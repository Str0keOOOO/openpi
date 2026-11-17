"""Parallel CPU gap search with persistent logs and per-case resume markers."""
import argparse,concurrent.futures,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from grouped import dump
HERE=Path(__file__).parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--resample',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);lock=(a.output/'scheduler.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);c=json.loads((a.dataset/'config.json').read_text());jobs=[(task,i) for i in range(5) for task in c['tasks']];began=time.time();total_gaps=json.loads((a.dataset/'sampling_summary.json').read_text())['invalid']
 dump(a.output/'config.json',dict(source=str(a.dataset),source_manifest_sha256=__import__('hashlib').sha256((a.dataset/'frozen_manifest.json').read_bytes()).hexdigest(),workers=a.workers,sobol_samples=256,differential_evolution_iterations=50,population_multiplier=5,sampling_mode='same_level_legal_resampling' if a.resample else 'original_stratum_search',changes_allowed='any legal point in the same scene, family and tier; sign/bin balance is a soft target' if a.resample else 'magnitudes inside the original signed sub-bins only',anchor_changes=False,scene_changes=False,task_success_filter=False,device='CPU'))
 def run(job):
  task,i=job;folder=a.output/'scenes'/task/f'scene_{i:02d}';folder.mkdir(parents=True,exist_ok=True)
  if (folder/'summary.json').exists() and json.loads((folder/'summary.json').read_text()).get('status')!='error':return
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONHASHSEED='0')
  with (folder/'run.log').open('a') as log:subprocess.run([sys.executable,'-u',str(HERE/'search_grouped_gaps_cpu.py'),'--dataset',str(a.dataset),'--output',str(a.output),'--task',task,'--scene',str(i)]+(['--resample'] if a.resample else []),env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
  futures=[pool.submit(run,j) for j in jobs]
  for future in concurrent.futures.as_completed(futures):
   future.result();summaries=[json.loads(p.read_text()) for p in a.output.glob('scenes/*/scene_*/summary.json')];cases=[json.loads(p.read_text()) for p in a.output.glob('scenes/*/scene_*/*/search_result.json')];progress=dict(total_gaps=total_gaps,processed=len(cases),legal_candidates=sum(x['status']=='legal_candidate_pending_official_export' for x in cases),not_found=sum(x['status']!='legal_candidate_pending_official_export' for x in cases),scenes_done=len(summaries),scene_errors=sum(x.get('status')=='error' for x in summaries),seconds=time.time()-began,device='CPU',policy_actions_executed=0);dump(a.output/'progress.json',progress);print(json.dumps(progress),flush=True)
 dump(a.output/'finished.json',progress)
if __name__=='__main__':main()
