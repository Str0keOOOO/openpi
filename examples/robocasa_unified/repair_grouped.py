"""Bounded physical-only same-band repair and final freeze of all test slots."""
import argparse,concurrent.futures,json,os,subprocess,sys,shutil,time
from pathlib import Path
from grouped import dump
from build_grouped import summarize
HERE=Path(__file__).parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,default=HERE/'grouped_config.json');p.add_argument('--gpus',nargs='+',type=int,required=True);a=p.parse_args();c=json.loads(a.config.read_text());r=a.output
 if (r/'FROZEN').exists():print('Already frozen; no resampling',flush=True);return
 if (r/'frozen_manifest.json').exists() and not (r/'initial_sampling_manifest.json').exists():(r/'frozen_manifest.json').rename(r/'initial_sampling_manifest.json')
 if (r/'finished.json').exists() and not (r/'initial_sampling_finished.json').exists():shutil.copy2(r/'finished.json',r/'initial_sampling_finished.json')
 uuids={g:subprocess.check_output(['nvidia-smi','-i',str(g),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip() for g in a.gpus};jobs=[(t,i) for i in range(5) for t in c['tasks']];started=time.time()
 def run(item):
  task,i,g=item;folder=r/'scenes'/task/f'scene_{i:02d}';env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuids[g],MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONHASHSEED='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',__EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/10_nvidia.json')
  with (folder/'quota_repair.log').open('a') as log:subprocess.run([sys.executable,'-u',str(HERE/'repair_grouped_scene.py'),'--config',str(a.config),'--output',str(r),'--task',task,'--scene',str(i),'--gpu',str(g)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
  return task,i
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for result in pool.map(run,[(task,i,a.gpus[j%len(a.gpus)]) for j,(task,i) in enumerate(jobs)]):print(result,flush=True)
 final=summarize(c,r,final=True);final.update(quota_repair_seconds=time.time()-started);dump(r/'finished.json',final);dump(r/'progress.json',final)
 # Final cases include physical-only re-pairing; keep the requested plan separately.
 if not (r/'plan_initial.json').exists():shutil.copy2(r/'plan.json',r/'plan_initial.json')
 manifest=json.loads((r/'frozen_manifest.json').read_text());dump(r/'plan.json',[x['case'] for x in manifest['cases']]);print(json.dumps(final),flush=True)
if __name__=='__main__':main()
