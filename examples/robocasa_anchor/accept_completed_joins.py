"""Install complete successful new rollouts, retaining failures and their exact reasons."""
import argparse
import collections
import concurrent.futures
import csv
import fcntl
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from retime_payload import build,write_json


def category(reason):
    if 'penetration' in reason or 'Illegal contact' in reason:return 'collision_or_penetration'
    if 'tracking tolerance' in reason or 'tracking error' in reason:return 'tracking_not_completed'
    if 'operation_failed' in reason:return 'task_failed_after_operation'
    if 'IK failed' in reason:return 'ik_failed'
    if 'path saving' in reason or 'path reduction' in reason:return 'screened_without_new_execution'
    if 'joint excursion' in reason:return 'joint_limit'
    return 'other'


def write_combined_results(dataset,source,work,results):
    with (source/'results.csv').open() as f:
        rows=list(csv.DictReader(f))
    for row in rows:
        new=results.get((row['task'],int(row['episode'])))
        if new and new['status']=='ready':
            row.update(status='ready',reason='',payload=new['payload'],
                       acceptance_policy='user_requested_completed_successful_new_rollout',
                       quality_warnings=json.dumps(new['quality_warnings']),
                       selected_trial=new['selected_trial'])
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (work/'combined_results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary=json.loads((dataset/'continuous_join_summary.json').read_text())
    assert sum(row['status']=='ready' for row in rows)==summary['replaced']
    summary.update(source_results_csv=str(source/'results.csv'),additional_results_csv=str(work/'results.csv'),
                   results_csv=str(work/'combined_results.csv'))
    write_json(dataset/'continuous_join_summary.json',summary)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    a.work.mkdir(parents=True,exist_ok=True)
    lock=(a.work/'scheduler.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (a.work/'finished.json').exists():
        print((a.work/'finished.json').read_text(),flush=True)
        return
    inventory=json.loads((a.source/'use_all_successful_new_inventory.json').read_text())
    assert len(inventory)==568
    source_summary=json.loads((a.dataset/'continuous_join_summary.json').read_text())
    write_json(a.work/'config.json',dict(dataset=str(a.dataset),source_inventory=str(a.source/'use_all_successful_new_inventory.json'),
        acceptance='actual operation success plus complete finite synchronized data and identical first state',
        waived='optimization quality thresholds retained as diagnostics, not rejection criteria',
        new_simulation=False,workers=a.workers))
    begin=time.time()
    results={}
    def absolute(path):
        q=Path(path)
        return q if q.is_absolute() else a.dataset.parent/q
    def process(item):
        task,idx=item['task'],item['episode']
        dest=a.work/'payloads'/f'{task}_ep{idx:06d}'
        status=a.work/'records'/f'{task}_ep{idx:06d}.json'
        if status.exists():
            saved=json.loads(status.read_text())
            if saved['status']!='ready' or (dest/'ready.json').exists():return saved
        old_prov=a.dataset/task/f'lerobot/extras/episode_{idx:06d}/provenance.json'
        human=json.loads(old_prov.read_text())['source_human_episode_index']
        base=dict(task=task,episode=idx,source_human_episode_index=human,
                  source_provenance_sha256=hashlib.sha256(old_prov.read_bytes()).hexdigest())
        good=item['new_successful_candidates']
        if not good:
            result=dict(base,status='kept_original',reason_category=category(item['old_rejection_reason']),
                        reason=item['old_rejection_reason'],attempts=item['attempts'])
        else:
            def rank(candidate):
                r=json.loads((absolute(candidate['path'])/'result.json').read_text())
                tr,op=r['transition'],r['operation']
                return (bool(tr.get('collision_review_required') or op.get('collision_review_required')),
                        tr['final_position_error_m'],tr['final_rotation_error_rad'],tr['steps'])
            errors=[]
            result=None
            for candidate in sorted(good,key=rank):
                trial=absolute(candidate['path'])
                try:
                    ready=build(a.dataset,task,idx,trial,dest,continuous_handoff=True,accept_completed_rollout=True)
                    result=dict(base,status='ready',payload=str(dest),selected_trial=str(trial),**{k:v for k,v in ready.items() if k not in ['task','episode']})
                    break
                except Exception:
                    errors.append(dict(trial=str(trial),error=traceback.format_exc()))
            if result is None:
                result=dict(base,status='kept_original',reason_category='payload_integrity_failed',reason=json.dumps(errors))
        write_json(status,result)
        return result
    def report(stage):
        vals=list(results.values())
        counts=collections.Counter(x['status'] for x in vals)
        write_json(a.work/'progress.json',dict(stage=stage,total=568,processed=len(vals),new_ready=counts['ready'],
            remaining_old=counts['kept_original'],seconds=time.time()-begin))
        fields=['task','episode','source_human_episode_index','status','reason_category','reason','selected_trial','frames','quality_warnings']
        for filename,subset in [('results.csv',vals),('remaining_failures.csv',[x for x in vals if x['status']!='ready']),
                                ('task_failed_episodes.csv',[x for x in vals if x.get('reason_category')=='task_failed_after_operation'])]:
            with (a.work/filename).open('w') as f:
                writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
                writer.writeheader()
                writer.writerows({**x,'quality_warnings':json.dumps(x.get('quality_warnings',[]))} for x in subset)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(process,inventory):
            results[(result['task'],result['episode'])]=result
            report('exporting')
            if len(results)%20==0:print('Exported',len(results),'of',len(inventory),flush=True)
    ready_count=sum(r['status']=='ready' for r in results.values())
    for r in results.values():
        p=a.dataset/r['task']/f"lerobot/extras/episode_{r['episode']:06d}/provenance.json"
        if hashlib.sha256(p.read_bytes()).hexdigest()!=r['source_provenance_sha256']:
            raise RuntimeError('Dataset changed during payload export')
    report('committing')
    from retime_batch import commit
    args=SimpleNamespace(dataset=a.dataset,work=a.work,joint_limit_policy='soft',
        stage_suffix='.accept_completed_staging',backup_suffix='.before_accept_completed_join',
        summary_name='continuous_join_accept_completed_summary.json',method='continuous_free_space_clock_with_source_OSC_feedforward')
    commit(args,results)
    source_summary.update(replaced=source_summary['replaced']+ready_count,kept_original=568-ready_count,
        optimization_quality_passed=1008,additional_completed_new_rollouts=ready_count,
        acceptance_policy='user_requested_all_complete_successful_new_rollouts',
        remaining_failures_csv=str(a.work/'remaining_failures.csv'),task_failed_episodes_csv=str(a.work/'task_failed_episodes.csv'))
    write_json(a.dataset/'continuous_join_summary.json',source_summary)
    # Read back every affected actual state; original first states must remain identical.
    verified=0
    for r in results.values():
        if r['status']!='ready':continue
        path=a.dataset/r['task']/f"lerobot/extras/episode_{r['episode']:06d}"
        before=a.dataset.with_name(a.dataset.name+'.before_accept_completed_join')/r['task']/f"lerobot/extras/episode_{r['episode']:06d}"
        with np.load(path/'states.npz') as x,np.load(before/'states.npz') as y:
            if not np.allclose(x['states'][0],y['states'][0],atol=1e-9,rtol=0):raise RuntimeError('First state changed after commit')
        verified+=1
    write_json(a.work/'verification.json',dict(passed=True,actual_first_states_verified=verified,
        additional_replacements=ready_count,total_new=1008+ready_count,remaining_old=568-ready_count,
        failure_counts=dict(collections.Counter(r.get('reason_category') for r in results.values() if r['status']!='ready'))))
    write_combined_results(a.dataset,a.source,a.work,results)
    report('finished')
    print(json.dumps(source_summary,indent=2),flush=True)


if __name__=='__main__':
    main()
