"""Generate real t→paired-anchor(0) labels offline, one episode per process.

Model is loaded once; each window has an independent MuJoCo segmentation capture.
Existing window NPZs are reused after their metadata/hash is checked by assembly.
"""
import argparse,json,os,shutil,subprocess,sys,traceback,tempfile
from pathlib import Path
import numpy as np
from examples.robocasa_dual_teacher.roma_scene import load_roma,process,checkpoint_sha256,TEACHER_ROOT
from examples.robocasa_dual_teacher.provenance import verify_sources
from examples.robocasa_dual_teacher.assemble_cache import assemble

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--task',required=True);p.add_argument('--episode',type=int,required=True);p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--frame',type=int,action='append');p.add_argument('--device',choices=['cuda','cpu'],default='cpu');p.add_argument('--position-scale-m',type=float,default=.1);p.add_argument('--rotation-scale-rad',type=float,default=1.);p.add_argument('--egl-device-id',type=int);a=p.parse_args()
 e=a.dataset/a.task/'lerobot/extras'/f'episode_{a.episode:06d}'
 with np.load(e/'states.npz',allow_pickle=False) as z:n=len(z['states'])
 frames=list(range(n)) if a.frame is None else sorted(set(a.frame))
 if any(x<0 or x>=n for x in frames):raise ValueError('Frame out of bounds')
 root=a.work/a.task/f'episode_{a.episode:06d}';root.mkdir(parents=True,exist_ok=True)
 model=predict=info=None;server=None;server_log=None;windows=[];errors=[]
 for t in frames:
  out=root/f'frame_{t:06d}';window=out/'window.npz';out.mkdir(exist_ok=True)
  try:
   if window.exists():
    try:
     with np.load(window,allow_pickle=False) as cached:
      m=json.loads(str(cached['metadata']))
      verify_sources(m,a.task,a.episode,a.dataset/a.task/'lerobot')
      if m.get('cache_version')!='roma_ee_window_v2_current_to_anchor' or m.get('ee_delta')!='anchor minus current; rotation log(R_anchor_root @ R_current_root.T)' or int(m['frame'])!=t or m['position_scale_m']!=a.position_scale_m or m['rotation_scale_rad']!=a.rotation_scale_rad or m['roma_checkpoint_sha256']!=checkpoint_sha256(TEACHER_ROOT/'weights/romav2.0.1.pt') or m.get('roma_seed')!=0 or m.get('mask_policy')!='robot_only_current_and_anchor_v2':raise ValueError('changed frame or settings')
      if cached['scene_latent'].shape!=(1,64) or cached['scene_valid'].shape!=(1,64) or cached['ee_latent'].shape!=(1,6) or cached['ee_valid'].shape!=(1,6):raise ValueError('invalid cached shapes')
    except Exception:
     # Preserve stale results for audit; do not mistake them for valid labels.
     import time
     stale=out/f'window.stale.{time.time_ns()}.npz';window.rename(stale)
   if not window.exists():
    if predict is None:model,predict,info=load_roma(a.device)
    with tempfile.TemporaryDirectory(prefix='dual_capture_',dir=out) as tmp:
     capture=Path(tmp)
     args=['--dataset',str(a.dataset),'--task',a.task,'--episode',str(a.episode),'--frame',str(t),'--output',str(capture),'--position-scale-m',str(a.position_scale_m),'--rotation-scale-rad',str(a.rotation_scale_rad)]
     if server is None:
      env=dict(os.environ)
      if a.egl_device_id is not None:
       env['CUDA_VISIBLE_DEVICES']=str(a.egl_device_id);env['MUJOCO_EGL_DEVICE_ID']=str(a.egl_device_id)
       env['MUJOCO_GL']='egl';env['PYOPENGL_PLATFORM']='egl'
      server_log=(root/'capture_server.log').open('a')
      server=subprocess.Popen([sys.executable,'-u','-m','examples.robocasa_dual_teacher.capture_server'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=server_log,text=True,bufsize=1,env=env)
     if server.poll() is not None:raise RuntimeError(f'Capture server exited {server.returncode}; see {root}/capture_server.log')
     server.stdin.write(json.dumps({'args':args})+'\n');server.stdin.flush();response=server.stdout.readline()
     if not response:raise RuntimeError('Capture server terminated without a response')
     response=json.loads(response)
     if not response['ok']:raise RuntimeError('capture failed: '+response.get('traceback',response['error'])[-1800:])
     process(capture,out,predict,model_info=info)
   # Assembly rechecks all source hashes before trusting this result.
   windows.append(window)
   (out/'error.json').unlink(missing_ok=True)
  except Exception as e:
   errors.append({'frame':t,'error':str(e)});(out/'error.json').write_text(json.dumps(errors[-1],indent=2))
  print(json.dumps({'task':a.task,'episode':a.episode,'frame':t,'done':len(windows),'failed':len(errors),'total':len(frames)}),flush=True)
  (root/'progress.json').write_text(json.dumps({'task':a.task,'episode':a.episode,'total':len(frames),'done':len(windows),'errors':errors},indent=2))
 if server is not None:
  server.stdin.close();server.wait(timeout=60);server_log.close()
 if windows:
  # Keep already completed windows when a partial --frame retry is requested.
  existing=sorted(root.glob('frame_*/window.npz'))
  output=assemble(existing,a.cache_root);print('assembled',output,flush=True)
 if errors:raise SystemExit('Some windows failed; inspect progress.json and rerun to retry')
if __name__=='__main__':main()
