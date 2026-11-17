"""Complete read-only checks and freeze the shared fixed-start manifest."""
import argparse,concurrent.futures,hashlib,json,os,subprocess,sys
from pathlib import Path
from grouped import dump
HERE=Path(__file__).parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gpus',type=int,nargs='+',required=True);a=p.parse_args();r=a.output;c=json.loads((r/'config.json').read_text());uuids={g:subprocess.check_output(['nvidia-smi','-i',str(g),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip() for g in a.gpus}
 def run(job):
  task,i,g=job;folder=r/'scenes'/task/f'scene_{i:02d}';env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuids[g],MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONHASHSEED='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',__EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/10_nvidia.json')
  with (folder/'validation.log').open('w') as log:subprocess.run([sys.executable,'-u',str(HERE/'validate_grouped_scene.py'),'--output',str(r),'--task',task,'--scene',str(i),'--gpu',str(g)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
  return json.loads((folder/'restoration_validation.json').read_text())
 jobs=[(task,i,a.gpus[j%len(a.gpus)]) for j,(task,i) in enumerate((t,i) for i in range(5) for t in c['tasks'])]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:checks=list(pool.map(run,jobs))
 summary=dict(passed=all(x['passed'] for x in checks),scenes_checked=len(checks),valid_starts_checked=sum(x.get('valid_starts_checked',0) for x in checks),manifest_sha256=hashlib.sha256((r/'frozen_manifest.json').read_bytes()).hexdigest(),policy_actions_executed=0,errors=[x['error'] for x in checks if not x['passed']]);dump(r/'validation_summary.json',summary)
 if summary['passed']:(r/'FROZEN').write_text(summary['manifest_sha256']+'\n')
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
