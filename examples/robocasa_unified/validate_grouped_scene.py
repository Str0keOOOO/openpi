"""Independently restore accepted fixed starts in a fresh scene, without stepping actions."""
import argparse,json,os,sys,traceback
from pathlib import Path
import numpy as np
from grouped import *

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--task',required=True);p.add_argument('--scene',type=int,required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args();directory=a.output/'scenes'/a.task/f'scene_{a.scene:02d}';env=None
 try:
  sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'robocasa_augmentation'));from gpu_routing import egl_device_for_uuid
  egl=egl_device_for_uuid(os.environ['CUDA_VISIBLE_DEVICES']);os.environ['MUJOCO_EGL_DEVICE_ID']=str(egl);os.environ['CUDA_VISIBLE_DEVICES']=str(egl)
  from sample_grouped_scene import make_env
  c=json.loads((a.output/'config.json').read_text());anchor=json.loads((directory/'anchor/anchor.json').read_text());env=make_env(a.task,anchor['scene'],anchor['scene']['seed'],egl)
  with np.load(directory/'anchor/snapshot.npz') as z:baseline={k:z[k].copy() for k in z.files}
  restore(env,anchor,baseline,reload_model=True);checks=[];records=[]
  for path in sorted((directory/'starts').glob('*/configuration.json')):
   r=json.loads(path.read_text())
   if not r['accepted']:continue
   restore(env,anchor,baseline);actual,result=apply(env,np.asarray(r['actual_delta']))
   with np.load(r['state_file']) as z:saved={k:z[k].copy() for k in z.files}
   delta=float(np.max(np.abs(env.sim.data.qpos-saved['qpos'])))
   if result['status']!='ready' or delta>1e-7 or not validate_delta(c,r['case'],actual):raise RuntimeError('Fixed start does not replay legally: '+str(path))
   ob=env._get_observations(force_update=True);state=np.concatenate([np.asarray(ob[k]) for k in STATE_KEYS])
   if not np.allclose(state,r['observation_state'],atol=1e-7,rtol=0):raise RuntimeError('State16 differs from saved same-frame observation')
   checks.append(dict(case=r['case']['id'],qpos_error=delta,actual_delta_error=float(np.max(np.abs(actual-r['actual_delta'])))));records.append(r)
  # Fresh XML + runtime model + actual perturbed state must reproduce RGB too.
  chosen=max(records,key=lambda r:np.max(np.abs(r['actual_delta'])/np.asarray(c['training_limits'])))
  with np.load(chosen['state_file']) as z:saved={k:z[k].copy() for k in z.files}
  restore(env,anchor,saved,reload_model=True);state,paths,poses=images_and_state(env,directory/'restoration_validation')
  import imageio.v2 as imageio
  maes={cam:float(np.abs(imageio.imread(chosen['images'][cam]).astype(float)-imageio.imread(paths[cam]).astype(float)).mean()) for cam in CAMERAS}
  if not np.allclose(state,chosen['observation_state'],atol=1e-7,rtol=0) or max(maes.values())>.5:raise RuntimeError('Independent perturbed RGB/State restoration differs: '+str(maes))
  dump(directory/'restoration_validation.json',dict(passed=True,valid_starts_checked=len(checks),checks=checks,perturbed_rgb_case=chosen['case']['id'],rgb_mae_255=maes,policy_actions_executed=0));print(a.task,a.scene,len(checks),'PASS',flush=True)
 except Exception:dump(directory/'restoration_validation.json',dict(passed=False,error=traceback.format_exc()));print(traceback.format_exc(),flush=True)
 finally:
  if env is not None:env.close()
if __name__=='__main__':main()
