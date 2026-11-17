"""Resume exact-start speed retiming; commit same-index replacements as a dataset transaction."""
import argparse,concurrent.futures,csv,fcntl,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
from retime_payload import build,write_json,CAMERAS
CODE=Path(__file__).parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--gpus',type=int,nargs='+',required=True);p.add_argument('--workers-per-gpu',type=int,default=4);p.add_argument('--joint-limit-policy',choices=['strict','soft'],default='strict');a=p.parse_args();a.work.mkdir(parents=True,exist_ok=True)
 lock=(a.work/'scheduler.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 gpu={}
 for g in a.gpus:
  q=subprocess.run(['nvidia-smi','-i',str(g),'--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True)
  if q.returncode==0 and q.stdout.strip().startswith('GPU-'):gpu[g]=q.stdout.strip()
 write_json(a.work/'config.json',dict(dataset=str(a.dataset),gpus=gpu,unavailable_gpus=[g for g in a.gpus if g not in gpu],max_total_gpu_memory_fraction=.9,workers_per_gpu=a.workers_per_gpu,joint_limit_policy=a.joint_limit_policy,soft_limits=dict(angular_cap_deg=.6,max_consecutive_frames=6,linear_tolerance_m=.001) if a.joint_limit_policy=='soft' else None,method='human_speed_continuous_smoothstep',initial_state='exact_saved_first_frame',replace_only_if='task success, initial equality, tracking tolerance, contact screening, not slower',originals='preserved by hardlink snapshot; atomic directory switch after batch'))
 if not gpu:raise RuntimeError('No available selected GPU')
 jobs=[]
 for f in a.dataset.glob('*/lerobot/extras/episode_*/provenance.json'):
  d=json.loads(f.read_text())
  if d.get('transition_frames',0)>0:
   jobs.append(dict(task=f.parents[3].name,episode=int(f.parent.name.split('_')[-1]),human=d['source_human_episode_index'],old_steps=d['transition_frames']))
 jobs.sort(key=lambda j:(j['task'],j['episode']));assert len(jobs)==1576
 records={};start=time.time();gpus=list(gpu)
 def execute(job,g):
  key=f"{job['task']}_ep{job['episode']:06d}";out=a.work/'trials'/key;payload=a.work/'payloads'/key;out.mkdir(parents=True,exist_ok=True);begin=time.time()
  if (payload/'ready.json').exists():return dict(job,status='ready',cached=True,payload=str(payload),**{k:v for k,v in json.loads((payload/'ready.json').read_text()).items() if k not in job})
  previous=out/'retime_status.json'
  if previous.exists():
   old=json.loads(previous.read_text())
   if a.joint_limit_policy=='strict' and old.get('status')=='kept_original' and 'joint-limit violation' in old.get('reason',''):
    return dict(old,cached=True)
  # Keep ten percent of this GPU free. No dummy allocation or other-process termination.
  while True:
   q=subprocess.run(['nvidia-smi','-i',str(g),'--query-gpu=memory.total,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True)
   try:
    total,free=map(int,q.stdout.strip().split(','))
    if free-total*.1>2200:break
   except ValueError:raise RuntimeError('GPU unavailable during run')
   time.sleep(5)
  env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=gpu[g],MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',__EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/10_nvidia.json')
  extra=a.dataset/job['task']/'lerobot/extras'/f"episode_{job['episode']:06d}"
  cmd=[sys.executable,'-u',str(CODE/('preflight_soft_limits.py' if a.joint_limit_policy=='soft' else 'preflight.py')),'--task',job['task'],'--episode',str(job['human']),'--gpu',str(g),'--output',str(out),'--execute-transition','--execute-operation','--reference-anchor-extras',str(extra),'--restore-initial-extras',str(extra),'--delta11',*(['0']*11)]
  try:
   with (out/'run.log').open('a') as log:
    q=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
   r=json.loads((out/'result.json').read_text())
   if q.returncode or r.get('status')!='operation_success':raise ValueError(r.get('failure_reason') or r.get('error') or r.get('transition',{}).get('failure_reason') or r.get('status'))
   ready=build(a.dataset,job['task'],job['episode'],out,payload)
   result=dict(job,status='ready',gpu=g,seconds=time.time()-begin,payload=str(payload),**{k:v for k,v in ready.items() if k not in job})
   # Keep only one synchronized payload, not duplicate 3-camera rollout videos/NPZ.
   for f in out.iterdir():
    if f.suffix in ('.mp4','.npz'):f.unlink()
  except Exception as ex:
   result=dict(job,status='kept_original',gpu=g,seconds=time.time()-begin,reason=str(ex)[-2500:])
  write_json(out/'retime_status.json',result);return result
 def progress(stage):
  vals=list(records.values());success=[r for r in vals if r['status']=='ready'];sec=[r['seconds'] for r in vals if 'seconds'in r and not r.get('cached')];elapsed=time.time()-start
  write_json(a.work/'progress.json',dict(stage=stage,total=len(jobs),processed=len(vals),ready=len(success),kept_original=len(vals)-len(success),gpus=list(gpu),joint_limit_policy=a.joint_limit_policy,reused=sum(bool(r.get('cached')) for r in vals),elapsed_seconds=elapsed,mean_worker_seconds=float(np.mean(sec)) if sec else None,throughput_per_hour=len(vals)/max(elapsed,1)*3600,estimated_remaining_hours=(len(jobs)-len(vals))*float(np.mean(sec))/(len(gpu)*a.workers_per_gpu)/3600 if sec else None))
  pd.DataFrame(vals).to_csv(a.work/'results.csv',index=False)
 # First independently validate one real sample in each task. No batch commit before this gate.
 pilot_ids={('OpenDrawer',489),('OpenStandMixerHead',191),('PickPlaceCounterToCabinet',154),('PickPlaceSinkToCounter',382)}
 pilots=[j for j in jobs if (j['task'],j['episode']) in pilot_ids]
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(pilots)) as pool:
  futures={pool.submit(execute,j,gpus[i%len(gpus)]):j for i,j in enumerate(pilots)}
  for future in concurrent.futures.as_completed(futures):
   j=futures[future];r=future.result();records[(j['task'],j['episode'])]=r;print(json.dumps(r),flush=True);progress('pilot_validation')
 # A particular episode may remain infeasible under the original joint/contact
 # checks. Preserve it, then require another verified pilot from that task.
 for task in sorted({j['task'] for j in jobs}):
  if any(r['task']==task and r['status']=='ready' for r in records.values()):continue
  alternatives=[j for j in jobs if j['task']==task and (j['task'],j['episode']) not in records]
  alternatives.sort(key=lambda j:abs(j['old_steps']-40))
  for j in alternatives[:3]:
   r=execute(j,gpus[len(records)%len(gpus)]);records[(j['task'],j['episode'])]=r
   print(json.dumps(r),flush=True);progress('pilot_validation')
   if r['status']=='ready':break
 if not all(any(r['task']==t and r['status']=='ready' for r in records.values()) for t in {j['task'] for j in jobs}):
  progress('pilot_failed_requires_fix');print('PILOT FAILED: original dataset unchanged',flush=True);return
 # Each GPU has its own bounded slot queue, preventing all jobs racing onto a single GPU.
 queues={g:[] for g in gpus};remaining=[j for j in jobs if (j['task'],j['episode']) not in records]
 for i,j in enumerate(remaining):queues[gpus[i%len(gpus)]].append(j)
 import queue
 qmap={g:queue.Queue() for g in gpus}
 for g,js in queues.items():
  for j in js:qmap[g].put(j)
 def slot(g):
  while True:
   try:j=qmap[g].get_nowait()
   except queue.Empty:return
   r=execute(j,g)
   with report_lock:
    records[(j['task'],j['episode'])]=r;print(json.dumps(r),flush=True);progress('generating')
 import threading
 report_lock=threading.Lock()
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)*a.workers_per_gpu) as pool:
  list(pool.map(slot,[g for g in gpus for _ in range(a.workers_per_gpu)]))
 progress('building_transaction');commit(a,records);progress('finished')

def commit(a,records):
 from lerobot.datasets.compute_stats import aggregate_stats,get_feature_stats
 from robocasa.utils import lerobot_utils as LU
 stage=a.dataset.with_name(a.dataset.name+getattr(a,'stage_suffix','.retime_staging'));backup=a.dataset.with_name(a.dataset.name+getattr(a,'backup_suffix','.before_retime'))
 if backup.exists():raise RuntimeError('Existing snapshot; refuse ambiguous second commit')
 if stage.exists():shutil.rmtree(stage)
 shutil.copytree(a.dataset,stage,copy_function=os.link,symlinks=True)
 def replace_file(src,dst):
  tmp=dst.with_name(dst.name+'.retime_tmp');shutil.copy2(src,tmp);os.replace(tmp,dst)
 for r in records.values():
  if r['status']!='ready':continue
  ds=stage/r['task']/'lerobot';idx=r['episode'];payload=Path(r['payload']);extra=ds/'extras'/f'episode_{idx:06d}'
  replace_file(payload/'episode.parquet',ds/'data/chunk-000'/f'episode_{idx:06d}.parquet')
  for name in ['states.npz','ee_world.npz','provenance.json','source_frame_mapping.npz']:
   if (payload/name).exists():replace_file(payload/name,extra/name)
  for c in CAMERAS:replace_file(payload/(c+'.mp4'),ds/'videos/chunk-000'/('observation.images.'+c)/f'episode_{idx:06d}.mp4')
 for td in stage.iterdir():
  ds=td/'lerobot'
  if not ds.exists():continue
  meta=ds/'meta';eps=[json.loads(x) for x in (meta/'episodes.jsonl').read_text().splitlines()];stats={x['episode_index']:x for x in map(json.loads,(meta/'episodes_stats.jsonl').read_text().splitlines())};offset=0
  for ep in eps:
   idx=ep['episode_index'];f=ds/'data/chunk-000'/f'episode_{idx:06d}.parquet';table=pq.read_table(f);n=len(table)
   indexes=np.arange(offset,offset+n,dtype=np.int64);table=table.set_column(table.schema.get_field_index('index'),'index',pa.array(indexes));tmp=f.with_name(f.name+'.tmp');pq.write_table(table,tmp);tmp.replace(f)
   r=records.get((td.name,idx))
   if r and r['status']=='ready':stats[idx]=json.loads((Path(r['payload'])/'episode_stats.json').read_text())
   stats[idx]['stats']['index']={k:v.tolist() for k,v in get_feature_stats(indexes,axis=0,keepdims=True).items()}
   ep['length']=n;offset+=n
  for name,rows in [('episodes.jsonl',eps),('episodes_stats.jsonl',[stats[i] for i in sorted(stats)])]:
   tmp=meta/(name+'.tmp');tmp.write_text(''.join(json.dumps(x)+'\n' for x in rows));tmp.replace(meta/name)
  info=json.loads((meta/'info.json').read_text());info['total_frames']=offset;write_json(meta/'info.json',info)
  # Project q01/q99 statistics use its existing official calculation helper.
  write_json(meta/'stats.json',LU.calculate_dataset_statistics([str(f) for f in (ds/'data').glob('*/episode_*.parquet')]))
 write_json(stage/getattr(a,'summary_name','transition_retime_summary.json'),dict(total_requested=len(records),replaced=sum(r['status']=='ready' for r in records.values()),kept_original=sum(r['status']!='ready' for r in records.values()),initial_states_unchanged=True,results_csv=str(a.work/'results.csv'),method=getattr(a,'method','human_speed_continuous_smoothstep'),joint_limit_policy=a.joint_limit_policy))
 # All initial pairings/episode IDs and filenames are unchanged. Keep an immutable old snapshot.
 a.dataset.rename(backup)
 try:stage.rename(a.dataset)
 except BaseException:backup.rename(a.dataset);raise
 write_json(a.work/'finished.json',dict(status='committed',dataset=str(a.dataset),original_snapshot=str(backup)))
if __name__=='__main__':main()
