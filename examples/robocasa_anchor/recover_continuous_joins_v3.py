"""Resume actual OSC recovery, retaining exact starts, successful sources and legality."""
import argparse,collections,concurrent.futures,csv,fcntl,hashlib,json,os,queue,shutil,subprocess,sys,threading,time,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from retime_payload import build,write_json
CODE=Path(__file__).parent
REVISION='successful_execution_reference_and_early_handoff_v3'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--gpus',type=int,nargs='+',required=True);p.add_argument('--workers-per-gpu',type=int,default=4);p.add_argument('--reuse-work',type=Path,required=True);p.add_argument('--pilot-only',action='store_true');a=p.parse_args()
 a.work.mkdir(parents=True,exist_ok=True);lock=(a.work/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (a.work/'finished.json').exists():print('Already finished');return
 jobs=list(csv.DictReader(a.selection.open()))
 for j in jobs:
  j['episode']=int(j['episode']);j['human']=int(j.get('source_human_episode_index',j.get('human')));j['key']=f"{j['task']}_ep{j['episode']:06d}";j['sha']=sha(a.dataset/j['task']/f"lerobot/extras/episode_{j['episode']:06d}/provenance.json")
 gpu={}
 for g in a.gpus:
  q=subprocess.run(['nvidia-smi','-i',str(g),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True);u,f=q.stdout.strip().split(',')
  if int(f)>8000:gpu[g]=u.strip()
 if not gpu:raise RuntimeError('No requested GPU has 8GB free')
 write_json(a.work/'config.json',dict(revision=REVISION,dataset=str(a.dataset),selection=str(a.selection),jobs=len(jobs),gpus=gpu,workers_per_gpu=a.workers_per_gpu,preview='diagnostic only, does not command actual arm joint waypoints',actual_acceptance='official task success; identical first state; recorded joint/self/environment contact checks; finite synchronized arrays/videos',minimum_optimization_gain=0,handoff_position_tolerance_m=.008,operation_max_time_ratio=1.4,reference_grasp_contact_margin_m=.001,source_posture_margin_rad=.035,reuse_work=str(a.reuse_work),source_human_unchanged=True,initial_configuration_unchanged=True))
 begin=time.time();records={};mutex=threading.Lock()
 def report(stage):
  vals=list(records.values());dur=[r['seconds'] for r in vals if not r.get('cached')];remaining=len(jobs)-len(vals)
  write_json(a.work/'progress.json',dict(stage=stage,total=len(jobs),processed=len(vals),recovered=sum(r['status']=='ready' for r in vals),remaining_old=sum(r['status']!='ready' for r in vals),elapsed_seconds=time.time()-begin,mean_worker_seconds=float(np.mean(dur)) if dur else None,estimated_remaining_hours=remaining*np.mean(dur)/(len(gpu)*a.workers_per_gpu*3600) if dur else None))
  fields=['task','episode','human','status','reason','seconds','payload','selected_trial','quality_warnings']
  for name,rows in [('results.csv',vals),('remaining_failures.csv',[r for r in vals if r['status']!='ready'])]:
   temp=a.work/(name+'.tmp')
   with temp.open('w') as f:w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
   temp.replace(a.work/name)
 def run(j,g):
  start=time.time();folder=a.work/'trials'/j['key'];folder.mkdir(parents=True,exist_ok=True);st=folder/'status.json';payload=a.work/'payloads'/j['key'];extra=a.dataset/j['task']/f"lerobot/extras/episode_{j['episode']:06d}"
  if st.exists():
   prev=json.loads(st.read_text())
   if prev['sha']!=j['sha']:raise RuntimeError('Source changed: '+j['key'])
   return dict(prev,cached=True)
  previous=a.reuse_work/'trials'/j['key']/'status.json'
  if previous.exists():
   cached=json.loads(previous.read_text())
   if cached['status']=='ready' and (Path(cached['payload'])/'ready.json').exists():
    if cached['sha']!=j['sha']:raise RuntimeError('Reuse source hash mismatch')
    cached.update(cached=True,reason='',reused_from=str(previous));write_json(st,cached);return cached
  attempts=[];result=None
  # First preserve the existing curve; then feedback, other joins and smooth avoidance arcs.
  # Reference the already successful same-start execution first. Preserve all
  # manipulation phases; only reconnect in the collision-free precontact tail.
  settings=[
   (0,1.0,1.6,.3,0.,True,4,.06,False,True,1.3),
   (0,1.15,1.6,.3,0.,True,8,.035,True,False,1.0),
   (0,1.2,2.0,.5,0.,True,8,.10,False,True,1.4),
   (2,1.1,1.7,.4,0.,True,4,.06,False,False,1.3),
   (0,1.35,2.0,.5,0.,False,8,.08,False,True,1.4),
   (0,1.0,1.6,.3,0.,True,0,.035,True,True,1.0),
   (0,1.2,2.3,.7,0.,False,12,.12,False,True,1.4),
   (0,1.4,2.0,.5,.02,True,10,.10,False,True,1.4),
   (3,1.2,1.8,.5,0.,True,6,.06,True,False,1.0),
   (0,1.5,2.5,.8,0.,False,16,.08,False,False,1.4),
   (0,1.0,1.6,.3,0.,True,6,.12,True,True,1.0),
   (0,1.5,2.0,.5,.04,True,12,.12,False,True,1.4),
  ]
  env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=gpu[g],MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',__EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/10_nvidia.json')
  try:
   for idx,(rank,scale,feedback,integral,lift,self_reference,tail_margin,posture_margin,source_only,guidance,time_ratio) in enumerate(settings):
    trial=folder/f'candidate_{idx}';trial.mkdir(exist_ok=True)
    cmd=[sys.executable,'-u',str(CODE/'preflight_continuous.py'),'--task',j['task'],'--episode',str(j['episode'] if self_reference else j['human']),'--gpu',str(g),'--output',str(trial),'--execute-transition','--execute-operation','--reference-anchor-extras',str(extra),'--restore-initial-extras',str(extra),'--delta11',*(['0']*11),'--handoff-rank',str(rank),'--handoff-duration-scale',str(scale),'--minimum-predicted-path-saving-m','0','--reference-phase-aligned','--ik-preview-diagnostic','--continuous-feedback-gain',str(feedback),'--continuous-integral-gain',str(integral),'--bridge-lift-m',str(lift)]
    cmd+=['--reference-grasp-contacts','--tracking-catchup','--tracking-max-time-ratio',str(time_ratio),'--max-interpolation-substeps','6','--handoff-position-tolerance','.008']
    if guidance:cmd.append('--source-posture-guidance')
    if source_only:cmd.append('--operation-source-actions')
    cmd+=['--handoff-tail-margin-frames',str(tail_margin),'--posture-limit-margin-rad',str(posture_margin)]
    if self_reference:cmd+=['--source-dataset',str(a.dataset)]
    done=False
    if (trial/'result.json').exists() and (trial/'command.json').exists():
     old=json.loads((trial/'command.json').read_text());r=json.loads((trial/'result.json').read_text());done=old['command']==cmd and r.get('status') not in ['running','ik_preflight_completed']
    if not done:
     write_json(trial/'command.json',dict(command=cmd,gpu_uuid=gpu[g],revision=REVISION))
     with (trial/'run.log').open('a') as log:
      try:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
      except subprocess.TimeoutExpired:attempts.append(dict(candidate=idx,status='timeout'));continue
    r=json.loads((trial/'result.json').read_text())
    if r.get('status')=='operation_success':
     try:
      if r['transition'].get('collision_review_required') or r['operation'].get('collision_review_required'):raise ValueError('Actual executed robot/environment contact requires review')
      if not r.get('soft_joint_limit_report',{}).get('accepted'):raise ValueError('Missing/pending executed joint-limit report')
      ready=build(a.dataset,j['task'],j['episode'],trial,payload,continuous_handoff=True,accept_completed_rollout=True)
      result=dict(j,status='ready',reason='',payload=str(payload),selected_trial=str(trial),quality_warnings=ready.get('quality_warnings',[]));attempts.append(dict(candidate=idx,status='accepted',ik_preview_valid=r.get('ik_preview_valid')));break
     except Exception as exc:reason=str(exc)
    else:reason=r.get('failure_reason') or r.get('error') or r.get('transition',{}).get('failure_reason') or r.get('status')
    attempts.append(dict(candidate=idx,status=r.get('status'),reason=str(reason)[-1800:]))
    write_json(folder/'attempts.json',attempts)
    # Failed outputs never enter the dataset; remove bulky failed trial videos/arrays after diagnosis.
    for f in trial.iterdir():
     if f.suffix in ['.mp4','.npz'] and f.name!='initial_state.npz':f.unlink()
    if r.get('status')=='anchor_initial_collision':break
   if result is None:result=dict(j,status='kept_original',reason=attempts[-1].get('reason','No completed valid candidate') if attempts else 'No candidate')
  except Exception:result=dict(j,status='kept_original',reason=traceback.format_exc())
  result.update(seconds=time.time()-start,cached=False,attempts=attempts);write_json(st,result)
  if result['status']=='ready':
   for f in Path(result['selected_trial']).iterdir():
    if f.suffix in ['.mp4','.npz']:f.unlink()
  return result
 def process(batch,stage):
  pending=queue.Queue()
  for j in batch:pending.put(j)
  def slot(g):
   while True:
    try:j=pending.get_nowait()
    except queue.Empty:return
    r=run(j,g)
    with mutex:records[(j['task'],j['episode'])]=r;report(stage);print(json.dumps({k:r.get(k) for k in ['task','episode','status','seconds','reason']}),flush=True)
  with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu)*a.workers_per_gpu) as pool:list(pool.map(slot,[g for g in gpu for _ in range(a.workers_per_gpu)]))
 # Representative categories are real rollouts, reused automatically in the full batch.
 pilots=[]
 for task,idx in [('OpenStandMixerHead',201),('PickPlaceCounterToCabinet',169),('PickPlaceSinkToCounter',128),('OpenDrawer',268)]:
  x=next((j for j in jobs if j['task']==task and j['episode']==idx),None)
  if x:pilots.append(x)
 report('pilot');process(pilots,'pilot')
 if a.pilot_only:return
 chosen={(j['task'],j['episode']) for j in pilots};process([j for j in jobs if (j['task'],j['episode']) not in chosen],'recovering')
 if not any(r['status']=='ready' for r in records.values()):report('no_recovery');return
 for j in jobs:
  if sha(a.dataset/j['task']/f"lerobot/extras/episode_{j['episode']:06d}/provenance.json")!=j['sha']:raise RuntimeError('Dataset changed during recovery')
 report('committing')
 from retime_batch import commit
 args=SimpleNamespace(dataset=a.dataset,work=a.work,joint_limit_policy='soft',stage_suffix='.recovery_v3_staging',backup_suffix='.before_continuous_join_recovery_v3',summary_name='continuous_join_recovery_v3_summary.json',method='continuous_free_space_clock_with_source_OSC_feedforward')
 commit(args,records)
 summary=json.loads((a.dataset/'continuous_join_summary.json').read_text());original=list(csv.DictReader(Path(summary['results_csv']).open()));n=0
 for row in original:
  new=records.get((row['task'],int(row['episode'])))
  if new and new['status']=='ready':row.update(status='ready',reason='',payload=new['payload'],selected_trial=new['selected_trial']);n+=1
 fields=list(dict.fromkeys(k for row in original for k in row))
 with (a.work/'combined_results.csv').open('w') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(original)
 summary.update(replaced=summary['replaced']+n,kept_original=summary['kept_original']-n,results_csv=str(a.work/'combined_results.csv'),remaining_failures_csv=str(a.work/'remaining_failures.csv'),recovery_count=n)
 write_json(a.dataset/'continuous_join_summary.json',summary)
 verified=0
 for r in records.values():
  if r['status']!='ready':continue
  rel=Path(r['task'])/f"lerobot/extras/episode_{r['episode']:06d}/states.npz"
  with np.load(a.dataset/rel) as new,np.load(a.dataset.with_name(a.dataset.name+'.before_continuous_join_recovery_v3')/rel) as old:
   if not np.array_equal(new['states'][0],old['states'][0]):raise RuntimeError('First state changed')
  verified+=1
 write_json(a.work/'verification.json',dict(passed=True,exact_initial_states=verified,recovered=n));report('finished')
if __name__=='__main__':main()
