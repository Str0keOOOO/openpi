"""Resumable four-GPU Anchor rollouts and serialized LeRobot export."""

from openpi.shared.paths import workspace_path
import argparse,concurrent.futures,csv,json,os,queue,signal,subprocess,sys,threading,time,traceback
from pathlib import Path


def dump(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False));tmp.replace(path)


def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=workspace_path('data/train'));p.add_argument('--dataset',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--gpus',type=int,nargs='+',default=[0,1,3,4]);p.add_argument('--timeout',type=int,default=900);p.add_argument('--retry-errors',action='store_true');p.add_argument('--episode-list',type=Path,help='CSV with task,episode columns; run only these source episodes');p.add_argument('--adaptive-interpolation',action='store_true',help='Use local EE interpolation for saturated targets');p.add_argument('--min-interpolation-substeps',type=int,default=2);p.add_argument('--max-interpolation-substeps',type=int,default=4);p.add_argument('--retry-operation-failures',action='store_true',help='Reexecute only existing operation failures; successful source episodes remain deduplicated');a=p.parse_args()
 if a.source.resolve()!=workspace_path('data/train').resolve(): raise ValueError('The current simulator worker is configured for the cleaned Human source only')
 from export_lerobot import initialize,rebuild_episode_map,append
 a.run_root.mkdir(parents=True,exist_ok=True);(a.run_root/'logs').mkdir(exist_ok=True)
 initialize(a.source,a.dataset);rebuild_episode_map(a.source,a.dataset)
 (a.run_root/'finished.txt').unlink(missing_ok=True)
 rows=list(csv.DictReader((a.source/'anchor_records.csv').open()));all_jobs=[(r['task'],int(r['filtered_episode_index'])) for r in rows]
 if len(all_jobs)!=418 or len(set(all_jobs))!=418:raise ValueError(f'Expected 418 unique source episodes, found {len(all_jobs)}')
 jobs=[(r['task'],int(r['episode'])) for r in csv.DictReader(a.episode_list.open())] if a.episode_list else all_jobs
 if not jobs or len(set(jobs))!=len(jobs) or any(j not in set(all_jobs) for j in jobs):raise ValueError(f'Invalid or duplicate episode list: {len(jobs)} jobs')
 if not 1 <= a.min_interpolation_substeps <= a.max_interpolation_substeps <= 8:raise ValueError('Require 1 <= --min-interpolation-substeps <= --max-interpolation-substeps <= 8')
 dump(a.run_root/'config.json',{'source':str(a.source),'dataset':str(a.dataset),'gpus':a.gpus,'jobs':len(jobs),'episode_list':str(a.episode_list) if a.episode_list else None,'per_source_limit':1,'controller':'PandaOmron OSC delta','tracking':'measured_ee_feedback_with_original_osc_feedforward','max_operation_time_ratio':1.30,'adaptive_interpolation':a.adaptive_interpolation,'min_interpolation_substeps':a.min_interpolation_substeps,'max_interpolation_substeps':a.max_interpolation_substeps,'selection':'exact CSV Anchor must pass RoboCasa official start-pose collision check; transition tracking and task success must pass; natural scene-object motion is recorded but does not reject the rollout','started_at':time.strftime('%Y-%m-%d %H:%M:%S')})
 slots=queue.Queue()
 for gpu in a.gpus:slots.put(gpu)
 export_queue=queue.Queue();mutex=threading.Lock();results={};export_state={}
 def persist():
  records=[]
  for task,ep in jobs:
   r=results.get((task,ep),{});ex=export_state.get((task,ep),{})
   op=r.get('operation') or {};anchor_collision=r.get('official_anchor_collision') or {}
   tr=r.get('transition') or {}
   reason=r.get('failure_reason') or r.get('error') or tr.get('failure_reason') or ex.get('error')
   if not reason and r.get('status')=='operation_failed':reason='task_condition_not_reached'
   records.append({'task':task,'episode':ep,'rollout_status':r.get('status','pending'),'stage':r.get('stage','pending'),'export_status':ex.get('status','pending'),'official_anchor_collision':anchor_collision.get('collision'),'anchor_contact_pairs':len(anchor_collision.get('contact_pairs',[])),'transition_deep_contact_events':len(tr.get('deep_contacts',[])),'operation_deep_contact_events':len(op.get('deep_contacts',[])),'failure':reason,'result_path':str(a.run_root/'episodes'/task/f'episode_{ep:06d}'/'result.json')})
  summary={'total':len(jobs),'rollout_finished':sum(x['rollout_status'] not in ('pending','running') for x in records),'rollout_success':sum(x['rollout_status']=='operation_success' for x in records),'exported':sum(x['export_status']=='exported' for x in records),'export_errors':sum(x['export_status']=='error' for x in records),'failures':sum(x['rollout_status'] not in ('pending','running','operation_success') for x in records),'running':sum(x['rollout_status']=='running' for x in records)}
  dump(a.run_root/'summary.json',summary)
  tmp=a.run_root/'results.csv.tmp'
  with tmp.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
  tmp.replace(a.run_root/'results.csv')
  print(json.dumps(summary),flush=True)
 def exporter():
  while True:
   job=export_queue.get()
   if job is None:export_queue.task_done();return
   task,ep=job;pilot=a.run_root/'episodes'/task/f'episode_{ep:06d}'
   try:
    append(a.source,a.dataset,task,ep,pilot);state={'status':'exported'}
   except Exception:state={'status':'error','error':traceback.format_exc()}
   with mutex:export_state[job]=state;persist()
   export_queue.task_done()
 export_thread=threading.Thread(target=exporter,daemon=False);export_thread.start()
 def run(job):
  task,ep=job;dest=a.run_root/'episodes'/task/f'episode_{ep:06d}';result=dest/'result.json';gpu=slots.get()
  try:
   existing=json.loads(result.read_text()) if result.exists() else {}
   terminal=('operation_success','operation_failed','transition_tracking_failed','anchor_initial_collision','error')
   if existing.get('status') not in terminal or (a.retry_errors and existing.get('status')=='error') or (a.retry_operation_failures and existing.get('status')=='operation_failed'):
    dest.mkdir(parents=True,exist_ok=True)
    with mutex:results[job]={'status':'running','stage':'initialization'};persist()
    cmd=[sys.executable,'-u',str(Path(__file__).with_name('preflight.py')),'--task',task,'--episode',str(ep),'--gpu',str(gpu),'--output',str(dest),'--execute-transition','--execute-operation','--min-interpolation-substeps',str(a.min_interpolation_substeps),'--max-interpolation-substeps',str(a.max_interpolation_substeps)]
    if a.adaptive_interpolation:cmd.append('--adaptive-interpolation')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',__EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/10_nvidia.json',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONPATH=os.environ.get('PYTHONPATH', ''))
    with (a.run_root/'logs'/f'{task}_{ep:06d}.log').open('a') as f:
     proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
     try:code=proc.wait(timeout=a.timeout)
     except subprocess.TimeoutExpired:
      os.killpg(proc.pid,signal.SIGTERM)
      try:proc.wait(timeout=10)
      except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
      code=-1
    existing=json.loads(result.read_text()) if result.exists() else {}
    if code or existing.get('status') not in ('operation_success','operation_failed','transition_tracking_failed','anchor_initial_collision','error'):
     existing.update(status='error',error=existing.get('error',f'worker exit {code} or missing terminal result'));dump(result,existing)
   with mutex:
    results[job]=existing
    if existing.get('status')=='operation_success':export_state[job]={'status':'queued'}
    else:export_state[job]={'status':'not_eligible'}
    persist()
   if existing.get('status')=='operation_success':export_queue.put(job)
  finally:slots.put(gpu)
 with mutex:persist()
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
  for fut in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):fut.result()
 export_queue.join();export_queue.put(None);export_thread.join()
 with mutex:persist()
 (a.run_root/'finished.txt').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))
if __name__=='__main__':main()
