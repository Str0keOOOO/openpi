"""Optimize screened handoffs, preserving exact starts and successful old trajectories."""
import argparse,concurrent.futures,csv,fcntl,hashlib,json,os,queue,shutil,subprocess,sys,threading,time,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from retime_payload import build,write_json
CODE=Path(__file__).parent

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,type=Path);p.add_argument('--work',required=True,type=Path);p.add_argument('--selection',required=True,type=Path);p.add_argument('--gpus',required=True,nargs='+',type=int);p.add_argument('--workers-per-gpu',type=int,default=4);a=p.parse_args();a.work.mkdir(parents=True,exist_ok=True)
 lock=(a.work/'scheduler.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (a.work/'finished.json').exists() and json.loads((a.work/'finished.json').read_text()).get('status')=='committed':print('Already committed; nothing to rerun',flush=True);return
 gpu={}
 for g in a.gpus:
  q=subprocess.run(['nvidia-smi','-i',str(g),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True)
  if q.returncode==0:
   uuid,free=q.stdout.strip().split(',');
   if int(free.strip())>=8000:gpu[g]=uuid.strip()
 if not gpu:raise RuntimeError('No selected GPU has sufficient free memory')
 jobs=list(csv.DictReader(a.selection.open()))
 for j in jobs:
  for k in ['episode','human','old_steps','old_total_frames']:j[k]=int(j[k])
  j['key']=f"{j['task']}_ep{j['episode']:06d}";j['input_provenance_sha256']=digest(a.dataset/j['task']/f"lerobot/extras/episode_{j['episode']:06d}/provenance.json")
 write_json(a.work/'config.json',dict(dataset=str(a.dataset),selection=str(a.selection),gpus=gpu,workers_per_gpu=a.workers_per_gpu,jobs=len(jobs),method='same_human_precontact_quintic_join',initial_state='exact actual saved first frame, unchanged',replacement_requirements=['official task success','actual joint/contact checks','same initial simulation state','new complete execution not slower','actual total EE path >=1cm and >=2% shorter','handoff direction not materially worse'],source_human_unchanged=True,backup='hardlink snapshot before atomic dataset directory switch',teacher_cache='Any old window-level teachers for replaced episodes must be regenerated; provenance and state changes invalidate source hashes'))
 records={};started=time.time();gpus=list(gpu);report_lock=threading.Lock();baseline_free={g:None for g in gpus}
 def progress(stage):
  vals=list(records.values());fresh=[r for r in vals if not r.get('cached') and r.get('seconds')];seconds=[r['seconds'] for r in fresh];elapsed=time.time()-started;remaining=len(jobs)-len(vals);slots=len(gpu)*a.workers_per_gpu
  write_json(a.work/'progress.json',dict(stage=stage,total=len(jobs),processed=len(vals),ready=sum(r['status']=='ready' for r in vals),kept_original=sum(r['status']!='ready' for r in vals),cached=sum(bool(r.get('cached')) for r in vals),gpus=gpus,workers_per_gpu=a.workers_per_gpu,elapsed_seconds=elapsed,mean_worker_seconds=float(np.mean(seconds)) if seconds else None,estimated_remaining_hours=remaining*float(np.mean(seconds))/slots/3600 if seconds else None,throughput_new_per_hour=len(fresh)/max(elapsed,1)*3600))
  if vals:
   # Keep nested diagnosis in JSONL, with a compact stable CSV index.
   temp=a.work/'results.jsonl.tmp';temp.write_text(''.join(json.dumps(r)+'\n' for r in vals));temp.replace(a.work/'results.jsonl')
   columns=['task','episode','human','status','gpu','seconds','cached','reason','payload','human_operation_start_frame']
   temp=a.work/'results.csv.tmp'
   with temp.open('w') as f:
    w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(vals)
   temp.replace(a.work/'results.csv')
 def memory_ready(g):
  while True:
   q=subprocess.run(['nvidia-smi','-i',str(g),'--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True)
   if q.returncode:raise RuntimeError('GPU unavailable')
   if int(q.stdout.strip())>2400:return
   time.sleep(5)
 def execute(j,g):
  begin=time.time();folder=a.work/'trials'/j['key'];folder.mkdir(parents=True,exist_ok=True);payload=a.work/'payloads'/j['key'];status_file=folder/'status.json'
  if status_file.exists():
   prev=json.loads(status_file.read_text())
   if prev.get('input_provenance_sha256')!=j['input_provenance_sha256']:raise ValueError('Input changed since previous attempt: '+j['key'])
   if prev.get('status')!='ready' or (payload/'ready.json').exists():return dict(prev,cached=True)
  attempts=[];result=None;extra=a.dataset/j['task']/f"lerobot/extras/episode_{j['episode']:06d}"
  try:
   memory_ready(g)
   env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=gpu[g],MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',__EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/10_nvidia.json')
   for rank in range(3):
    trial=folder/f'candidate_{rank}';trial.mkdir(exist_ok=True);scale=1.65 if j['task']=='OpenDrawer' else 1.0
    cmd=[sys.executable,'-u',str(CODE/'preflight_smooth_join.py'),'--task',j['task'],'--episode',str(j['human']),'--gpu',str(g),'--output',str(trial),'--execute-transition','--execute-operation','--reference-anchor-extras',str(extra),'--restore-initial-extras',str(extra),'--delta11',*(['0']*11),'--handoff-rank',str(rank),'--handoff-duration-scale',str(scale),'--minimum-predicted-path-saving-m','.02']
    write_json(trial/'command.json',dict(command=cmd,gpu_uuid=gpu[g]))
    with (trial/'run.log').open('a') as log:
     try:q=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
     except subprocess.TimeoutExpired:attempts.append(dict(rank=rank,status='timeout'));continue
    if not (trial/'result.json').exists():attempts.append(dict(rank=rank,status='missing_result',exit_code=q.returncode));continue
    r=json.loads((trial/'result.json').read_text())
    if r.get('status')=='operation_success':
     try:
      ready=build(a.dataset,j['task'],j['episode'],trial,payload,smooth_handoff=True)
      result=dict(j,status='ready',gpu=g,seconds=time.time()-begin,payload=str(payload),attempts=attempts+[dict(rank=rank,status='accepted')],**{k:v for k,v in ready.items() if k not in j});break
     except Exception as ex:attempts.append(dict(rank=rank,status='quality_rejected',reason=str(ex)))
    else:
     reason=r.get('failure_reason') or r.get('error') or r.get('transition',{}).get('failure_reason') or r.get('status');attempts.append(dict(rank=rank,status=r.get('status'),reason=str(reason)[-1500:]))
     if r.get('status') in ['no_predicted_improvement','anchor_initial_collision']:break
     if 'No such handoff candidate rank' in str(reason):break
   if result is None:result=dict(j,status='kept_original',gpu=g,seconds=time.time()-begin,reason=attempts[-1].get('reason',attempts[-1]['status']) if attempts else 'No candidate',attempts=attempts)
  except Exception:result=dict(j,status='kept_original',gpu=g,seconds=time.time()-begin,reason=traceback.format_exc()[-2500:],attempts=attempts)
  write_json(status_file,result)
  if result['status']=='ready':
   # Complete synchronized payload is retained. Remove redundant temporary
   # renderings/states only after validation and payload creation succeeded.
   for trial in folder.glob('candidate_*'):
    for f in trial.iterdir():
     if f.suffix in ['.mp4','.npz']:f.unlink()
  return result
 # Validate/export the four already executed real pilots using the same payload builder.
 known=json.loads(Path('/data5/xuchenfei/handoff_comparison/final_selection.json').read_text())
 for item in known:
  source=Path(item['source']);s=json.loads((source/'summary.json').read_text());j=next((j for j in jobs if j['task']==s['task'] and j['episode']==s['episode']),None)
  if j is None:continue
  folder=a.work/'trials'/j['key'];folder.mkdir(parents=True,exist_ok=True);payload=a.work/'payloads'/j['key'];st=folder/'status.json'
  if st.exists():continue
  rank=next(x['rank'] for x in s['attempts'] if x['status']=='accepted');trial=source/f'candidate_{rank}'
  actual=json.loads((source/'actual_states_validation.json').read_text())
  if not actual['passed']:raise RuntimeError('Known pilot validation failed')
  ready=build(a.dataset,j['task'],j['episode'],trial,payload,smooth_handoff=True)
  prov=json.loads((payload/'provenance.json').read_text());prov['soft_joint_limit_report']=actual['joint_limits'];prov['soft_joint_limit_report']['policy']['scope']='combined actual transition and operation states';write_json(payload/'provenance.json',prov)
  r=dict(j,status='ready',gpu=None,seconds=s['seconds'],payload=str(payload),cached=True,source_pilot=str(source),**{k:v for k,v in ready.items() if k not in j});write_json(st,r)
 queues={g:queue.Queue() for g in gpus}
 # Interleave tasks to provide representative early throughput and failures.
 groups={t:[j for j in jobs if j['task']==t] for t in sorted({j['task'] for j in jobs})};ordered=[]
 for i in range(max(map(len,groups.values()))):
  for js in groups.values():
   if i<len(js):ordered.append(js[i])
 for i,j in enumerate(ordered):queues[gpus[i%len(gpus)]].put(j)
 def slot(g):
  while True:
   try:j=queues[g].get_nowait()
   except queue.Empty:return
   try:r=execute(j,g)
   except Exception:r=dict(j,status='kept_original',gpu=g,reason=traceback.format_exc()[-2500:])
   with report_lock:
    records[(j['task'],j['episode'])]=r;print(json.dumps({k:v for k,v in r.items() if k not in ['attempts','handoff_quality']}),flush=True);progress('generating')
 progress('starting')
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)*a.workers_per_gpu) as pool:list(pool.map(slot,[g for g in gpus for _ in range(a.workers_per_gpu)]))
 if not any(r['status']=='ready' for r in records.values()):progress('no_improvements');return
 progress('building_transaction')
 from retime_batch import commit
 args=SimpleNamespace(dataset=a.dataset,work=a.work,joint_limit_policy='soft',stage_suffix='.smooth_join_staging',backup_suffix='.before_smooth_join',summary_name='smooth_join_summary.json',method='same_human_precontact_quintic_join')
 commit(args,records);progress('finished')

if __name__=='__main__':main()
