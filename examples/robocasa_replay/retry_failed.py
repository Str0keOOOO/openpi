"""Repeat failed saved Human trajectories, stopping each episode after success."""
import argparse
import concurrent.futures
import csv
import datetime
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpus', type=int, nargs='+', required=True)
    p.add_argument('--per-gpu', type=int, default=2)
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--timeout', type=int, default=1200)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    logs = a.output / 'logs'
    logs.mkdir(exist_ok=True)
    replay = Path(__file__).resolve().with_name('replay_human.py')
    jobs = []
    for path in sorted((a.source / 'episodes').glob('*/episode_*/result.json')):
        row = json.loads(path.read_text())
        if row.get('status') == 'completed' and not row.get('success_ever'):
            jobs.append(dict(task=row['task'], episode_index=row['episode_index'],
                             source_episode_index=row.get('source_episode_index'),
                             original_result=str(path)))
    if not jobs:
        raise RuntimeError('No completed unsuccessful episodes found')
    write_json(a.output / 'jobs.json', jobs)
    write_json(a.output / 'run_config.json', dict(source=str(a.source), dataset=str(a.dataset),
               gpus=a.gpus, simulators_per_gpu=a.per_gpu, max_new_attempts_per_episode=a.repeats,
               stop_after='first new replay with success_ever=true', seed=0,
               perturbations_added=False, original_data_modified=False,
               started_at=datetime.datetime.now().astimezone().isoformat(), command=sys.argv))
    slots = queue.Queue()
    for _ in range(a.per_gpu):
        for gpu in a.gpus:
            slots.put(gpu)
    lock = threading.Lock()
    attempts = {}
    statuses = { (j['task'], j['episode_index']):dict(j, status='pending', attempts=0) for j in jobs }
    started = time.monotonic()

    def save():
        # Called with the lock held, including during concurrent completions.
        rows = sorted(attempts.values(), key=lambda r:(r['task'],r['episode_index'],r['repeat']))
        fields = ['task','episode_index','source_episode_index','repeat','gpu','status',
                  'success_ever','success_final','first_success_seconds','video','result','log','error']
        tmp = a.output / 'attempts.csv.tmp'
        with tmp.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
        tmp.replace(a.output / 'attempts.csv')
        state = list(statuses.values())
        summary = dict(planned_episodes=len(jobs), attempted_replays=len(rows),
                       recovered_episodes=sum(r['status']=='success' for r in state),
                       exhausted_episodes=sum(r['status']=='exhausted' for r in state),
                       running_episodes=sum(r['status']=='running' for r in state),
                       pending_episodes=sum(r['status']=='pending' for r in state),
                       elapsed_seconds=time.monotonic()-started, episodes=state)
        write_json(a.output / 'summary.json', summary)
        print(json.dumps({k:v for k,v in summary.items() if k!='episodes'}),flush=True)

    def run(job):
        key = (job['task'],job['episode_index'])
        gpu = slots.get()
        try:
            with lock:
                statuses[key].update(status='running',gpu=gpu); save()
            for repeat in range(1,a.repeats+1):
                root = a.output / 'attempts' / f'repeat_{repeat:02d}'
                dest = root / 'episodes' / job['task'] / f"episode_{job['episode_index']:06d}"
                result = dest / 'result.json'
                log = logs / f"{job['task']}_ep{job['episode_index']:06d}_repeat{repeat:02d}.log"
                old = json.loads(result.read_text()) if result.exists() else {}
                if old.get('status') == 'completed':
                    row = old
                else:
                    command = [sys.executable,'-u',str(replay),'--dataset',str(a.dataset),
                               '--output',str(root),'--task',job['task'],'--episode',str(job['episode_index']),
                               '--gpu',str(gpu),'--video-skip','2']
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu))
                    with log.open('a') as stream:
                        proc = subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,
                                                env=env,start_new_session=True)
                        try:
                            code=proc.wait(timeout=a.timeout)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid,signal.SIGTERM)
                            try: proc.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(proc.pid,signal.SIGKILL); proc.wait()
                            code=-1
                    row=json.loads(result.read_text()) if result.exists() else {}
                    if code or row.get('status')!='completed':
                        row.update(status='error',error=row.get('error',f'child exit code {code}'))
                record=dict(row,task=job['task'],episode_index=job['episode_index'],
                            source_episode_index=job['source_episode_index'], repeat=repeat,
                            gpu=row.get('gpu',gpu),result=str(result),log=str(log))
                success=record.get('status')=='completed' and bool(record.get('success_ever'))
                with lock:
                    attempts[(*key,repeat)]=record
                    statuses[key].update(attempts=repeat,last_result=str(result))
                    if success:
                        statuses[key].update(status='success',successful_repeat=repeat,video=record.get('video'))
                    save()
                if success: return
            with lock:
                statuses[key]['status']='exhausted'; save()
        finally:
            slots.put(gpu)

    with lock: save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.gpus)*a.per_gpu) as pool:
        futures=[pool.submit(run,j) for j in jobs]
        for f in concurrent.futures.as_completed(futures): f.result()
    with lock: save()
    write_json(a.output / 'finished.json', dict(finished_at=datetime.datetime.now().astimezone().isoformat()))


if __name__=='__main__':
    main()
