"""CPU coverage scheduler with isolated, one-per-GPU MuJoCo rollout workers."""
import argparse
import collections
import concurrent.futures
import csv
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from state import NAMES, SCALE, TOL, measure_pair, delta
from candidate_screen import screen_candidates

ANCHOR_CODE = Path(__file__).resolve().parents[1]/'robocasa_anchor'
HIGH_TARGETS = [('4', 60), ('5-7', 60), ('8-11', 35)]
sys.path.insert(0,str(ANCHOR_CODE))
import export_lerobot as exporter


def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def kind(values):
    n=int(np.sum(np.abs(values)>TOL))
    return ('zero' if n==0 else 'single' if n==1 else 'multi'), n


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--gpus',type=int,nargs='+',default=[0,1,3,4,6])
    parser.add_argument('--max-attempts',type=int,default=20000)
    parser.add_argument('--workers-per-gpu', type=int, default=4)
    parser.add_argument('--worker-memory-mib', type=int, default=4096)
    parser.add_argument('--gpu-reserve-mib', type=int, default=3072)
    parser.add_argument('--max-stalled-trials', type=int, default=50)
    parser.add_argument('--min-free-gib', type=int, default=40)
    parser.add_argument('--duration-hours', type=float, default=9.0)
    a=parser.parse_args();a.work.mkdir(parents=True,exist_ok=True)
    if a.duration_hours < 0: raise ValueError('duration-hours cannot be negative')
    deadline = time.time() + a.duration_hours * 3600 if a.duration_hours else None
    # One central writer; do not allow another scheduler to append concurrently.
    import fcntl
    lock=(a.work/'scheduler.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    db=sqlite3.connect(a.work/'generation.sqlite');db.execute('pragma journal_mode=WAL')
    db.execute('create table if not exists refs (task text,human integer,payload text,primary key(task,human))')
    db.execute('create table if not exists trials (id integer primary key,task text,human integer,anchor integer,gpu integer,requested text,actual text,status text,reason text,seconds real,episode integer)')
    db.execute('create table if not exists candidate_screens (id integer primary key,task text,human integer,requested text,reason text,contacts text)')
    db.execute("update trials set status='interrupted' where status='running'");db.commit()
    children={}
    def stop(signum,frame):
        for proc in list(children.values()):
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    atomic(a.work/'config.json',dict(source=str(a.source),output=str(a.output),gpus=a.gpus,
        names=NAMES,units=['m','m','deg','m']+['deg']*7,scale=SCALE.tolist(),tolerances=TOL.tolist(),
        single_per_dimension_cap=12,multi_per_task=dict(HIGH_TARGETS),
        planning='Human and all generated samples contribute to signed amplitude density; explicit higher-dimensional generation fills missing combinations.',
        workers_per_gpu=a.workers_per_gpu,worker_memory_mib=a.worker_memory_mib,gpu_reserve_mib=a.gpu_reserve_mib,
        max_stalled_trials=a.max_stalled_trials,min_free_gib=a.min_free_gib,
        duration_hours=a.duration_hours,deadline_unix=deadline,
        collision_rule='Official exact-start predicate; reject transition/operation collision_review_required; no task-contact blanket rejection',
        sample_source='Original Human trajectory; reference=actual paired Anchor first frame',max_attempts=a.max_attempts))
    pairs=pd.read_csv(a.source/'human_anchor_pairs.csv')
    assert len(pairs)==418 and not pairs.duplicated(['task','human_episode_index']).any()
    mapping=pd.read_csv(a.source/'episode_index_map.csv')
    assert (mapping.data_type=='human').sum()==418 and (mapping.data_type=='anchor').sum()==418
    if not a.output.exists():
        atomic(a.work/'progress.json',dict(stage='copy_immutable_source'))
        stage=a.output.with_name(a.output.name+'.building')
        if stage.exists():shutil.rmtree(stage)
        shutil.copytree(a.source,stage);stage.rename(a.output)
    journal=a.work/'export_transaction.json'
    if journal.exists():
        j=json.loads(journal.read_text());ds=a.output/j['task']/'lerobot';idx=j['episode'];backup=a.work/'meta_rollback';extra=ds/'extras'/f'episode_{idx:06d}'
        committed=False
        if (extra/'provenance.json').exists():
            try:committed=json.loads((extra/'provenance.json').read_text()).get('generation_trial_id')==j['trial_id']
            except ValueError:pass
        if not committed:
            for f in backup.iterdir():shutil.copy2(f,ds/'meta'/f.name)
            for f in (ds/'data').glob(f'*/episode_{idx:06d}.parquet'):f.unlink()
            for f in (ds/'videos').glob(f'*/*/episode_{idx:06d}.mp4'):f.unlink()
            shutil.rmtree(extra,ignore_errors=True)
        shutil.rmtree(backup,ignore_errors=True);journal.unlink()
    for f in a.output.glob('*/lerobot/extras/episode_*/provenance.json'):
        d=json.loads(f.read_text())
        if d.get('generation_trial_id'):
            db.execute('update trials set status=?,episode=? where id=?',('exported_'+d['data_type'].split('_',1)[1],d['episode_index'],d['generation_trial_id']))
    db.commit()
    existing={(x[0],x[1]) for x in db.execute('select task,human from refs')}
    jobs=[(str(a.source),r.task,int(r.human_episode_index),int(r.anchor_episode_index)) for r in pairs.itertuples() if (r.task,int(r.human_episode_index)) not in existing]
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(5,len(a.gpus))) as pool:
        for ref in pool.map(measure_pair,jobs,chunksize=1):
            db.execute('insert or replace into refs values(?,?,?)',(ref['task'],ref['human'],json.dumps(ref)));db.commit()
            atomic(a.work/'progress.json',dict(stage='audit_actual_first_frames',processed=db.execute('select count(*) from refs').fetchone()[0],total=418))
    refs=[json.loads(x[0]) for x in db.execute('select payload from refs order by task,human')]
    assert len(refs)==418
    tasks=sorted({r['task'] for r in refs})
    triples=sorted({tuple(sorted((shift%11,(shift+i)%11,(shift+j)%11))) for i,j in [(1,2),(1,3),(1,4),(1,5),(2,4)] for shift in range(11)})
    assert len(triples)==55
    records=[]
    for r in refs:records.append(dict(task=r['task'],human=r['human'],anchor=r['anchor'],data_type='human',values=r['human_delta']))
    for row in db.execute("select task,human,anchor,actual,status from trials where status like 'exported_%'"):
        records.append(dict(task=row[0],human=row[1],anchor=row[2],values=json.loads(row[3]),data_type='perturb_'+row[4].split('_',1)[1]))
    pending={};finished_attempts=db.execute('select count(*) from trials').fetchone()[0];pilot_required=not db.execute("select count(*) from trials where status like 'exported_%'").fetchone()[0]
    pilot_done=0;pilot_success=sum(r['data_type'].startswith('perturb_') for r in records)
    screen_pool = concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn'))
    histories = collections.defaultdict(lambda: collections.Counter())
    tried = set()
    failed_patterns = collections.Counter()
    def pattern_key(task, human, v):
        values=np.asarray(v);ids=tuple(np.flatnonzero(np.abs(values)>TOL).tolist())
        return task,human,tuple((i,1 if values[i]>0 else -1,min(2,int(abs(values[i])/SCALE[i]*3))) for i in ids)
    def request_key(task, human, v):
        return task, human, tuple(np.round(np.asarray(v) / SCALE * 100).astype(int))
    for task, human, requested, status in db.execute('select task,human,requested,status from trials'):
        histories[task,human]['success' if status.startswith('exported_') else 'failure'] += 1
        tried.add(request_key(task, human, json.loads(requested)))
        if not status.startswith('exported_'):
            failed_patterns[pattern_key(task,human,json.loads(requested))]+=1
    for task, human, requested in db.execute('select task,human,requested from candidate_screens'):
        tried.add(request_key(task, human, json.loads(requested)))
    gpu_uuids = {}
    gpu_resources = {}
    for gpu in a.gpus:
        probe = subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
        if probe.returncode == 0: gpu_uuids[gpu] = probe.stdout.strip()
    if not gpu_uuids: raise RuntimeError('None of the selected GPUs is healthy')
    stalled = 0
    stop_reason = 'attempt_budget'
    plan_round = 0

    def bucket(n):
        return '4' if n == 4 else '5-7' if 5 <= n <= 7 else '8-11' if n >= 8 else str(n)

    def higher_coverage():
        # Human tiny joint initialization noise remains in density counts, but
        # does not substitute for intentionally varied higher-dimensional data.
        result = collections.Counter()
        for r in records + [p['record'] for p in pending.values()]:
            if r.get('data_type') == 'human': continue
            n = kind(r['values'])[1]
            if n >= 4: result[r['task'],bucket(n)] += 1
        return result

    def counts(include_pending=True):
        singles=collections.Counter();groups=collections.Counter();cells=collections.Counter();combos=collections.Counter()
        items=records+([p['record'] for p in pending.values()] if include_pending else [])
        for r in items:
            v=np.asarray(r['values']);typ,n=kind(v);ids=tuple(np.flatnonzero(np.abs(v)>TOL).tolist());task=r['task']
            if n==1:
                singles[ids[0]]+=1;singles[(task,ids[0])]+=1
            if n>=2:groups[(task,min(n,4))]+=1;combos[(task,ids)]+=1
            for i in ids:
                if abs(v[i])>SCALE[i]+TOL[i] or (i==3 and v[i]<-TOL[i]):continue
                cells[(task,i,1 if v[i]>0 else -1,min(2,int(abs(v[i])/SCALE[i]*3)))]+=1
        return singles,groups,cells,combos

    def plan(gpu):
        nonlocal plan_round
        plan_round += 1
        singles,groups,cells,combos=counts()
        high = higher_coverage()
        slots=[]
        for t in tasks:
            for i in range(11):
                if singles[i]<12 and singles[t,i]<3:slots.append((singles[i]/12,t,'single',i))
            for group,target in HIGH_TARGETS:
                if high[t,group]<target:slots.append((high[t,group]/target,t,'multi',group))
        if not slots:return None
        trial_id=db.execute('select coalesce(max(id),0)+1 from trials').fetchone()[0]
        rng=np.random.default_rng(20261111+trial_id+7919*plan_round)
        rng.shuffle(slots);slots.sort(key=lambda x:x[0])
        for _,task,typ,key in slots[:4]:
            candidates=[r for r in refs if r['task']==task and sum(histories[task,r['human']].values())<80]
            if not candidates: continue
            weights=np.asarray([(histories[task,r['human']]['success']+1)/(histories[task,r['human']]['failure']+2)/math.sqrt(1+histories[task,r['human']]['success']) for r in candidates])
            selected=rng.choice(len(candidates),size=min(3,len(candidates)),replace=False,p=weights/weights.sum())
            for index in selected:
                ref=candidates[index]
                lo=np.asarray(ref['lo']);hi=np.asarray(ref['hi'])
                vectors=[]
                for _ in range(16):
                    n=1 if typ=='single' else 4 if key=='4' else int(rng.integers(5,8)) if key=='5-7' else int(rng.integers(8,12))
                    if typ=='single': ids=(key,)
                    else:
                        choices=[tuple(sorted(rng.choice(11,size=n,replace=False).tolist())) for _ in range(8)]
                        ids=min(choices,key=lambda ids:combos[task,ids])
                    v=np.zeros(11);feasible=True
                    for i in ids:
                        options=[]
                        for sign in ([1] if i==3 else [-1,1]):
                            for level in range(3):
                                low=level/3*SCALE[i];high=(level+1)/3*SCALE[i]
                                if sign>0:low=max(low,lo[i],TOL[i]*2);high=min(high,hi[i])
                                else:low=max(low,-hi[i],TOL[i]*2);high=min(high,-lo[i])
                                if high>low:
                                    # Single distribution also needs empty signed bins.
                                    count=cells[task,i,sign,level]
                                    if typ=='single':
                                        count=sum(1 for r in records if kind(r['values'])[1]==1 and (r['values'][i]>0)==(sign>0) and abs(r['values'][i])>TOL[i] and min(2,int(abs(r['values'][i])/SCALE[i]*3))==level)
                                    options.append((count,sign,low,high))
                        if not options:feasible=False;break
                        rng.shuffle(options);options.sort(key=lambda x:x[0])
                        # Mostly fill the least populated cell, sometimes explore
                        # its next neighbour rather than repeating an infeasible edge.
                        choice=options[0 if rng.random()<.8 else min(1,len(options)-1)]
                        _,sign,low,high=choice;v[i]=sign*rng.uniform(low,high)
                    k=request_key(task,ref['human'],v)
                    if not feasible or kind(v)[1]!=len(ids) or k in tried or failed_patterns[pattern_key(task,ref['human'],v)]>=2: continue
                    tried.add(k);vectors.append(v.tolist())
                if not vectors:continue
                try:
                    screens=screen_pool.submit(screen_candidates,str(a.source),ref,vectors).result(timeout=180)
                except Exception as exc:
                    db.execute('insert into candidate_screens(task,human,requested,reason,contacts) values(?,?,?,?,?)',(task,ref['human'],json.dumps(vectors[0]),'screen_error',str(exc)[-1000:]))
                    db.commit();print(f'CPU screen error {task}/{ref["human"]}: {exc}',flush=True)
                    continue
                chosen=None
                for vector,screen in zip(vectors,screens):
                    db.execute('insert into candidate_screens(task,human,requested,reason,contacts) values(?,?,?,?,?)',(task,ref['human'],json.dumps(vector),screen['reason'],json.dumps(screen.get('contacts',[]))))
                    if chosen is None and screen['valid']: chosen=vector
                db.commit()
                if chosen is None:continue
                v=np.asarray(chosen)
                record=dict(task=task,human=ref['human'],anchor=ref['anchor'],values=v.tolist())
                p=dict(id=trial_id,gpu=gpu,ref=ref,record=record,requested_type=typ)
                db.execute('insert into trials(id,task,human,anchor,gpu,requested,status) values(?,?,?,?,?,?,?)',(trial_id,task,ref['human'],ref['anchor'],gpu,json.dumps(v.tolist()),'running'));db.commit()
                return p
        return None

    def worker(p):
        start=time.time();out=a.work/'temporary'/f"trial_{p['id']:07d}";out.mkdir(parents=True,exist_ok=True)
        ref=p['ref'];v=p['record']['values'];reason='';result={}
        env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=gpu_uuids[p['gpu']],MUJOCO_EGL_DEVICE_ID=str(p['gpu']),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',__EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/10_nvidia.json')
        for scale in [1.]:
            requested=(np.asarray(v)*scale).tolist()
            cmd=[sys.executable,'-u',str(ANCHOR_CODE/'preflight.py'),'--task',ref['task'],'--episode',str(ref['human']),'--gpu',str(p['gpu']),'--output',str(out),'--execute-transition','--execute-operation','--reference-anchor-extras',str(a.source/ref['task']/'lerobot/extras'/f"episode_{ref['anchor']:06d}"),'--delta11',*[f'{x:.12f}' for x in requested]]
            with (out/'run.log').open('a') as log:
                proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children[p['id']]=proc
                try:code=proc.wait(timeout=900)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL);proc.wait();code=-1
                finally:children.pop(p['id'],None)
            f=out/'result.json';result=json.loads(f.read_text()) if f.exists() else dict(status='error',error='No worker result')
            if code:reason=f'Worker exit {code}: '+str(result.get('error',''));break
            if result.get('status')=='operation_success':break
            if result.get('status') in ['operation_failed','transition_tracking_failed']:break
        status=result.get('status','error')
        accepted=status=='operation_success' and result.get('operation',{}).get('success_ever') is True
        if accepted and (result.get('transition',{}).get('collision_review_required') or result.get('operation',{}).get('collision_review_required')):
            accepted=False;status='collision_rejected';reason='Illegal/deep robot contact flagged during executed trajectory'
        actual=None
        if accepted:
            z=np.load(out/'initial_state.npz');q=z['base_world_quat_wxyz'];yaw=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
            pose=np.r_[z['base_world_xyz'],yaw,z['torso_qpos'][0],z['arm_qpos']]
            with np.load(out/'transition_actual.npz') as tr:
                if not np.allclose(tr['sim_state_before_action'][0],z['sim_state'],atol=1e-9,rtol=0):accepted=False;reason='Executed first state differs from measured candidate state'
            actual=delta(pose,np.asarray(ref['anchor_pose']))
            if np.any(np.abs(actual)>SCALE+TOL) or actual[3]<-TOL[3] or kind(actual)[0]!=p['requested_type']:
                accepted=False;reason='Actual perturbation outside limits or changes Single/Multi class'
        if not reason:reason=result.get('failure_reason') or result.get('error') or status
        return dict(p=p,out=out,accepted=accepted,actual=actual,reason=str(reason)[-5000:],status=status,seconds=time.time()-start)

    def export(result):
        p=result['p'];ref=p['ref'];typ=kind(result['actual'])[0];ds=a.output/ref['task']/'lerobot';meta=ds/'meta';idx=json.loads((meta/'info.json').read_text())['total_episodes']
        backup=a.work/'meta_rollback'
        if backup.exists():raise RuntimeError('Unresolved export rollback exists; restore before resuming')
        backup.mkdir()
        atomic(a.work/'export_transaction.json',dict(task=ref['task'],episode=idx,trial_id=p['id']))
        for f in meta.iterdir():shutil.copy2(f,backup/f.name)
        try:
            exporter.append_in_place(a.source,a.output,ref['task'],ref['human'],result['out'],data_type='perturb_'+typ,refresh_stats=False)
            extra=ds/'extras'/f'episode_{idx:06d}'
            for name in ['provenance.json','ep_meta.json']:
                f=extra/name;d=json.loads(f.read_text());d.update(type='perturb_'+typ,data_type='perturb_'+typ,source_human_episode_index=ref['human'],reference_anchor_episode_index=ref['anchor'],generation_trial_id=p['id']);d.pop('source_pilot',None);atomic(f,d)
            # Verify the actual saved first frame, not just the requested offset.
            with np.load(extra/'states.npz') as z, np.load(result['out']/'initial_state.npz') as initial:
                if not np.allclose(z['states'][0],initial['sim_state'],atol=1e-9,rtol=0):raise ValueError('Exported first state mismatch')
        except Exception:
            for f in backup.iterdir():shutil.copy2(f,meta/f.name)
            for f in (ds/'data').glob(f'*/episode_{idx:06d}.parquet'):f.unlink()
            for f in (ds/'videos').glob(f'*/*/episode_{idx:06d}.mp4'):f.unlink()
            shutil.rmtree(ds/'extras'/f'episode_{idx:06d}',ignore_errors=True)
            raise
        finally:
            shutil.rmtree(backup)
            (a.work/'export_transaction.json').unlink(missing_ok=True)
        record=dict(p['record'],values=result['actual'].tolist(),data_type='perturb_'+typ);records.append(record)
        return typ,idx

    def report():
        rows=[]
        for r in records:
            v=np.asarray(r['values']);typ,n=kind(v)
            rows.append(dict(task=r['task'],human_episode_index=r['human'],reference_anchor_episode_index=r['anchor'],data_type=r['data_type'],derived_type=typ,active_dimensions=n,**dict(zip(NAMES,v))))
        pd.DataFrame(rows).to_csv(a.work/'actual_distribution.csv',index=False)
        pd.read_sql_query('select * from trials',db).to_csv(a.work/'attempts.csv',index=False)
        singles,groups,cells,combos=counts(False)
        counts_by_type=dict(collections.Counter(r['data_type'] for r in records));counts_by_type['anchor']=418
        atomic(a.work/'progress.json',dict(stage='generating',counts=counts_by_type,attempts=finished_attempts,pilot_success=pilot_success,inflight=len(pending),single_coverage={NAMES[i]:singles[i] for i in range(11)},multi_coverage={f'{t}:{n}':groups[t,n] for t in tasks for n in [2,3,4]},gpus=a.gpus))
        atomic(a.work/'scheduler_status.json',dict(time=time.time(),deadline_unix=deadline,stalled_trials=stalled,
            gpu_resources=gpu_resources,
            higher_dimensional_coverage={f'{t}:{b}':higher_coverage()[t,b] for t in tasks for b in ['4','5-7','8-11']},
            cpu_screen_outcomes=dict(db.execute('select reason,count(*) from candidate_screens group by reason')),
            pending=[dict(trial=p['id'],gpu=p['gpu'],task=p['ref']['task'],human=p['ref']['human']) for p in pending.values()]))
        # Rebuild all four type labels and links from saved episode provenance.
        rows=[]
        for task in tasks:
            for f in sorted((a.output/task/'lerobot/extras').glob('episode_*/provenance.json')):
                d=json.loads(f.read_text());h=d['source_filtered_episode_index'];typ=d['data_type'];idx=int(f.parent.name.split('_')[-1]);anchor=next(r['anchor'] for r in refs if r['task']==task and r['human']==h)
                rows.append(dict(task=task,filtered_episode_index=idx,source_episode_index=d['source_original_episode_index'],data_type=typ,type=typ,source_filtered_episode_index=h,source_human_episode_index=h,reference_anchor_episode_index=anchor))
        with (a.output/'episode_index_map.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        atomic(a.output/'augmentation_summary.json',dict(counts=counts_by_type,attempts=finished_attempts,distribution=str(a.work/'actual_distribution.csv'),attempt_statistics=str(a.work/'attempts.csv')))

    report()
    gpu_cooldowns={}
    def gpu_capacity(gpu):
        if time.time()<gpu_cooldowns.get(gpu,0):return 0
        try:
            # Numeric NVML selection avoids enumerating the failed GPU3 while
            # resolving a UUID. CUDA itself still binds the explicit UUID.
            probe=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=12)
            if probe.returncode:return 0
            free=int(probe.stdout.strip())
            owned={proc.pid for proc in list(children.values())}
            query=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=12)
            own_used=0
            for line in query.stdout.splitlines():
                parts=line.split(',')
                if len(parts)==2 and parts[0].strip().isdigit() and int(parts[0]) in owned and parts[1].strip().isdigit():own_used+=int(parts[1])
            capacity=max(0,min(a.workers_per_gpu,(free+own_used-a.gpu_reserve_mib)//a.worker_memory_mib))
            gpu_resources[gpu]=dict(time=time.time(),free_mib=free,owned_used_mib=own_used,capacity=capacity)
            return capacity
        except (ValueError,subprocess.TimeoutExpired):return 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.gpus)*a.workers_per_gpu) as pool:
        futures={};idle_rounds=0
        while not a.max_attempts or finished_attempts<a.max_attempts or futures:
            low_disk=bool(a.min_free_gib) and shutil.disk_usage(a.output).free<a.min_free_gib*2**30
            deadline_reached=deadline is not None and time.time()>=deadline
            stalled_limit=bool(a.max_stalled_trials) and stalled>=a.max_stalled_trials
            attempt_limit=bool(a.max_attempts) and finished_attempts+len(futures)>=a.max_attempts
            stop_launch=low_disk or deadline_reached or stalled_limit or attempt_limit
            if low_disk:stop_reason='disk_reserve';atomic(a.work/'resource_pause.json',dict(reason=f'Less than {a.min_free_gib} GiB free; draining jobs',time=time.time()))
            elif deadline_reached:stop_reason='time_limit'
            elif stalled_limit:stop_reason='no_success_guard'
            if not stop_launch:
                for gpu in gpu_uuids:
                    if deadline is not None and time.time()>=deadline:break
                    capacity=gpu_capacity(gpu)
                    active=sum(p['gpu']==gpu for p in pending.values())
                    for _ in range(max(0,capacity-active)):
                        if deadline is not None and time.time()>=deadline:break
                        p=plan(gpu)
                        if p is None:break
                        if deadline is not None and time.time()>=deadline:
                            db.execute("update trials set status='cancelled_deadline' where id=?",(p['id'],));db.commit()
                            break
                        pending[p['id']]=p;futures[pool.submit(worker,p)]=p
            if not futures:
                singles,_,_,_=counts(False);high=higher_coverage()
                missing=sum(max(0,12-singles[i]) for i in range(11))+sum(max(0,target-high[t,b]) for t in tasks for b,target in HIGH_TARGETS)
                if deadline is not None and time.time()>=deadline:stop_launch=True;stop_reason='time_limit'
                if stop_launch or not missing:stop_reason=stop_reason if stop_launch else 'coverage_complete';break
                idle_rounds+=1;report()
                time.sleep(60 if idle_rounds>=10 else 15);continue
            idle_rounds=0
            done,_=concurrent.futures.wait(futures,timeout=10,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                p=futures.pop(future);pending.pop(p['id']);result=future.result();idx=None;status=result['status']
                if result['accepted']:
                    db.execute('update trials set actual=?,status=? where id=?',(json.dumps(result['actual'].tolist()),'validated',p['id']));db.commit()
                    typ,idx=export(result);status='exported_'+typ;pilot_success+=1
                    stalled=0
                else:
                    stalled+=1
                    failed_patterns[pattern_key(p['ref']['task'],p['ref']['human'],p['record']['values'])]+=1
                histories[p['ref']['task'],p['ref']['human']]['success' if result['accepted'] else 'failure']+=1
                if any(term in result['reason'].lower() for term in ['out of memory','egl_bad_alloc','device lost','fallen off']):gpu_cooldowns[p['gpu']]=time.time()+300
                db.execute('update trials set actual=?,status=?,reason=?,seconds=?,episode=? where id=?',(json.dumps(result['actual'].tolist()) if result['actual'] is not None else None,status,result['reason'],result['seconds'],idx,p['id']));db.commit()
                # These are newly created transient files, never original data.
                shutil.rmtree(result['out']);finished_attempts+=1;pilot_done+=1
                report()
                print(f'trial={p["id"]} GPU={p["gpu"]} {p["record"]["task"]} Human={p["record"]["human"]} {status}',flush=True)
    screen_pool.shutdown()
    report()
    from robocasa.utils import lerobot_utils as LU
    for task in tasks:
        ds=a.output/task/'lerobot';atomic(ds/'meta/stats.json',LU.calculate_dataset_statistics([str(p) for p in (ds/'data').glob('*/episode_*.parquet')]))
    subprocess.run(['/home/xuchenfei/anaconda3/bin/python',str(Path(__file__).with_name('report.py')),'--work',str(a.work)],check=True,env=dict(os.environ,MPLCONFIGDIR='/tmp/augmentation_mpl'))
    singles,groups,_,_=counts(False);high=higher_coverage()
    missing=sum(max(0,12-singles[i]) for i in range(11))+sum(max(0,target-high[t,b]) for t in tasks for b,target in HIGH_TARGETS)
    atomic(a.work/'finished.json',dict(status='coverage_complete' if missing==0 else 'incomplete_coverage',reason=stop_reason,missing_slots=missing,attempts=finished_attempts))
    progress=json.loads((a.work/'progress.json').read_text());progress.update(stage='stopped',stop_reason=stop_reason,missing_slots=missing);atomic(a.work/'progress.json',progress)


if __name__=='__main__':main()
