"""Resumable offline teacher preparation over selected/all episodes, no training."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--egl-device-id',type=int,required=True);p.add_argument('--task',action='append');p.add_argument('--episode',type=int,action='append');a=p.parse_args()
 a.work.mkdir(parents=True,exist_ok=True);jobs=[]
 for f in sorted(a.dataset.glob('*/lerobot/extras/episode_*/provenance.json')):
  task=f.parents[3].name;ep=int(f.parent.name.split('_')[-1])
  if a.task and task not in a.task:continue
  if a.episode and ep not in a.episode:continue
  jobs.append((task,ep))
 if not jobs:raise ValueError('No dataset episodes matched')
 records=[]
 for i,(task,ep) in enumerate(jobs):
  log=a.work/task/f'episode_{ep:06d}.log';log.parent.mkdir(parents=True,exist_ok=True)
  cmd=[sys.executable,'-m','examples.robocasa_dual_teacher.generate_episode','--dataset',str(a.dataset),'--task',task,'--episode',str(ep),'--cache-root',str(a.cache_root),'--work',str(a.work/'windows'),'--device',a.device,'--egl-device-id',str(a.egl_device_id)]
  start=time.time()
  with log.open('a') as f:q=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
  records.append(dict(task=task,episode=ep,status='complete' if q.returncode==0 else 'failed',seconds=time.time()-start,log=str(log)))
  temp=a.work/'progress.tmp';temp.write_text(json.dumps(dict(total=len(jobs),processed=i+1,failed=sum(r['status']=='failed' for r in records),records=records),indent=2));temp.replace(a.work/'progress.json')
  print(json.dumps(records[-1]),flush=True)
 if any(r['status']=='failed' for r in records):raise SystemExit('Some episodes failed; rerun for resume')
if __name__=='__main__':main()
