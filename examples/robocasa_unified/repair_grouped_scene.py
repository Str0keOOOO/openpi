"""Re-pair same-tier dimension strata using physical legality only; preserve marginal quotas."""
import argparse,copy,json,os,sys,time,shutil
from pathlib import Path
import numpy as np
from grouped import *

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--task',required=True);p.add_argument('--scene',type=int,required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--rounds',type=int,default=160);a=p.parse_args();c=json.loads(a.config.read_text());directory=a.output/'scenes'/a.task/f'scene_{a.scene:02d}';env=None;began=time.time();report=[]
 try:
  sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'robocasa_augmentation'));from gpu_routing import egl_device_for_uuid
  egl=egl_device_for_uuid(os.environ['CUDA_VISIBLE_DEVICES']);os.environ['MUJOCO_EGL_DEVICE_ID']=str(egl);os.environ['CUDA_VISIBLE_DEVICES']=str(egl)
  from sample_grouped_scene import make_env
  anchor=json.loads((directory/'anchor/anchor.json').read_text())
  with np.load(directory/'anchor/snapshot.npz') as z:arrays={k:z[k].copy() for k in z.files}
  env=make_env(a.task,anchor['scene'],anchor['scene']['seed'],egl);restore(env,anchor,arrays,reload_model=True)
  def check(case,delta):
   restore(env,anchor,arrays);actual,result=apply(env,delta)
   if not validate_delta(c,case,actual) or not np.allclose(actual,delta,atol=1e-7,rtol=0):result['status']='invalid_actual_stratum'
   return result['status']=='ready',actual,result
  for family,active in FAMILIES.items():
   for level in ['small','medium','large']:
    name=family+'_'+level;marker=directory/'quota_repair'/name/'summary.json'
    if marker.exists():report.append(json.loads(marker.read_text()));continue
    marker.parent.mkdir(parents=True,exist_ok=True);records=[json.loads((directory/'starts'/f'{name}_{i:02d}'/'configuration.json').read_text()) for i in range(8)]
    initial=sum(r['accepted'] for r in records)
    if initial==8:summary=dict(family=family,level=level,initial_valid=8,final_valid=8,rounds=0);dump(marker,summary);report.append(summary);continue
    cases=[copy.deepcopy(r['case']) for r in records];values=[np.asarray(r.get('requested_delta',r.get('actual_delta')),float) for r in records];valid=[r['accepted'] for r in records];best=(sum(valid),copy.deepcopy(cases),[x.copy() for x in values],valid.copy());rng=rng_for(c['seed'],a.task,a.scene,family,level,'quota_repair');rounds=0
    for round_index in range(a.rounds):
     bad=[i for i,v in enumerate(valid) if not v]
     if not bad:break
     i=int(rng.choice(bad));j=int(rng.choice([k for k in range(8) if k!=i]));ca,cb=copy.deepcopy(cases[i]),copy.deepcopy(cases[j]);columns=rng.choice(active,int(rng.integers(1,len(active)+1)),replace=False)
     for axis in columns:
      for key in ['signs','magnitude_subbins']:ca[key][axis],cb[key][axis]=cb[key][axis],ca[key][axis]
     va=propose(c,ca,1000+round_index*2);vb=propose(c,cb,1001+round_index*2);oka,aa,ra=check(ca,va);okb,ab,rb=check(cb,vb);before=int(valid[i])+int(valid[j]);after=int(oka)+int(okb);take=after>=before or rng.random()<.025*(1-round_index/a.rounds)
     with (marker.parent/'attempts.jsonl').open('a') as f:f.write(json.dumps(dict(round=round_index,rows=[i,j],columns=columns.tolist(),legal=[oka,okb],status=[ra['status'],rb['status']],accepted_move=bool(take)))+'\n')
     if take:cases[i],cases[j]=ca,cb;values[i],values[j]=va,vb;valid[i],valid[j]=oka,okb
     if sum(valid)>best[0]:best=(sum(valid),copy.deepcopy(cases),[x.copy() for x in values],valid.copy())
     rounds=round_index+1
     if best[0]==8:break
    count,bc,bv,bok=best
    if count>initial:
     # Swapping column strata within the same scene preserves every signed
     # magnitude quota, including contributions from still-infeasible slots.
     for axis in active:
      old=sorted((r['case']['signs'][axis],r['case']['magnitude_subbins'][axis]) for r in records);new=sorted((x['signs'][axis],x['magnitude_subbins'][axis]) for x in bc)
      if old!=new:raise RuntimeError('Marginal sampling quota changed')
     for i in range(8):
      dest=directory/'starts'/bc[i]['id'];backup=directory/'initial_sampling'/bc[i]['id']
      if not backup.exists():backup.parent.mkdir(exist_ok=True);shutil.copytree(dest,backup,copy_function=os.link)
      ok,actual,legality=check(bc[i],bv[i]);record=dict(case=bc[i],accepted=ok,actual_delta=actual.tolist(),requested_delta=bv[i].tolist(),anchor=str(directory/'anchor/anchor.json'),scene=anchor['scene'],instruction=anchor['instruction'],legality=legality,policy_actions_executed=0,quota_repair=True,initial_accepted=records[i]['accepted'])
      if ok:
       # Replace files rather than mutate hardlinked audit copies.
       tmp=dest/'state.repair.npz';np.savez_compressed(tmp,**snapshot(env));tmp.replace(dest/'state.npz')
       state,paths,cam=images_and_state(env,dest/'repaired_images');record.update(state_file=str(dest/'state.npz'),observation_state=state.tolist(),images=paths,cameras=cam)
      else:record['reason']=legality['status']
      dump(dest/'configuration.json',record)
    summary=dict(family=family,level=level,initial_valid=initial,final_valid=count,rounds=rounds,unchanged_marginal_quotas=True,task_success_used=False);dump(marker,summary);report.append(summary)
  allrows=[json.loads(p.read_text()) for p in (directory/'starts').glob('*/configuration.json')];old=json.loads((directory/'summary.json').read_text());old.update(valid=sum(r['accepted'] for r in allrows),invalid=sum(not r['accepted'] for r in allrows),quota_repair=report);dump(directory/'summary.json',old);dump(directory/'quota_repair_summary.json',dict(status='completed',groups=report,seconds=time.time()-began));print(a.task,a.scene,old['valid'],old['invalid'],flush=True)
 except Exception:import traceback;dump(directory/'quota_repair_summary.json',dict(status='error',error=traceback.format_exc()));print(traceback.format_exc(),flush=True)
 finally:
  if env is not None:env.close()
if __name__=='__main__':main()
