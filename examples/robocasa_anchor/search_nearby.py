"""Search nearby successful Anchor candidates for unpaired Human episodes."""
import argparse,concurrent.futures,csv,json,math,os,signal,subprocess,sys,threading,time
from pathlib import Path
import cv2
import numpy as np
from nearby_candidates import RADII_M,YAW_DEG,YAW_LEVER_M

LOCK=threading.Lock()


def read_result(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=Path('/data4/xuchenfei/robocasa365/data/household_augmented'))
    p.add_argument('--dataset',type=Path,default=Path('/data5/xuchenfei/robocasa365_household_anchor_official_collision'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=5)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--max-attempts',type=int,default=28)
    p.add_argument('--successes-per-episode',type=int,default=2)
    p.add_argument('--episodes-file',type=Path)
    p.add_argument('--candidate-file',type=Path)
    p.add_argument('--previous-output',type=Path)
    p.add_argument('--dense-shortlist',action='store_true')
    p.add_argument('--directional-shortlist',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    mapping=list(csv.DictReader((a.dataset/'episode_index_map.csv').open()))
    paired={(r['task'],int(r['source_filtered_episode_index'])) for r in mapping if r['data_type']=='anchor'}
    jobs=sorted((r['task'],int(r['filtered_episode_index'])) for r in mapping if r['data_type']=='human' and (r['task'],int(r['filtered_episode_index'])) not in paired)
    if a.episodes_file:
        requested={(r['task'],int(r['episode'])) for r in csv.DictReader(a.episodes_file.open())}
        jobs=[j for j in jobs if j in requested]
        if set(jobs)!=requested:raise ValueError('Requested episode missing or already paired')
    config={**{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'source':str(a.source),'dataset':str(a.dataset),'output':str(a.output),'episodes':len(jobs),
            'radii_m':RADII_M,'yaw_deg':YAW_DEG,'yaw_distance_lever_m':YAW_LEVER_M,
            'pose':'only base SE2 changes; official nominal arm, torso, gripper retained',
            'ranking':'sampled SE2 distance; closest successful among tested candidates, not certified global optimum',
            'selection':'collect successful candidates for human choice; do not append automatically'}
    (a.output/'config.json').write_text(json.dumps(config,indent=2))
    with (a.output/'episodes.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['task','episode']);w.writerows(jobs)
    progress={j:'pending' for j in jobs};records={}
    def publish():
        # Call with LOCK held. Atomic summaries remain readable during processing.
        rows=[r for values in records.values() for r in values]
        cols=['task','episode','candidate_id','local_dx_m','local_dy_m','delta_yaw_deg','translation_m','distance_score_m','status','success','operation_steps','time_ratio','result_path','video','preview','failure']
        tmp=a.output/'candidates.csv.tmp'
        with tmp.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
        tmp.replace(a.output/'candidates.csv')
        success=[r for r in rows if r['success']]
        summary={'total_episodes':len(jobs),'episode_states':{f'{t}/EP{e}':v for (t,e),v in progress.items()},
                 'episodes_with_success':len({(r['task'],r['episode']) for r in success}),
                 'successful_candidates':len(success),'full_rollouts_recorded':len(rows),'updated_at':time.strftime('%F %T')}
        tmp=a.output/'summary.json.tmp';tmp.write_text(json.dumps(summary,indent=2));tmp.replace(a.output/'summary.json')
        # Image sheet is kept small and includes only genuinely successful runs.
        tiles=[]
        for row in sorted(success,key=lambda r:(r['task'],r['episode'],r['distance_score_m'])):
            im=cv2.imread(row['preview']) if row['preview'] else None
            if im is None:continue
            tile=np.full((218,512,3),255,np.uint8);tile[46:]=cv2.resize(im,(512,172))
            cv2.putText(tile,f"{row['task']} EP{row['episode']}",(5,16),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
            label=f"dx {100*row['local_dx_m']:+.2f}cm dy {100*row['local_dy_m']:+.2f}cm yaw {row['delta_yaw_deg']:+.1f}deg"
            cv2.putText(tile,label,(5,36),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1);tiles.append(tile)
        if tiles:
            while len(tiles)%3:tiles.append(np.full_like(tiles[0],255))
            cv2.imwrite(str(a.output/'successful_candidates.jpg'),np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)]))
        print(json.dumps({k:v for k,v in summary.items() if k!='episode_states'}),flush=True)
    def run_preflight(task,ep,dest,offset=None,screen=False):
        terminal=('screen_completed','operation_success','operation_failed','anchor_initial_collision','transition_tracking_failed','error')
        old=read_result(dest/'result.json')
        if old.get('status') in terminal and old.get('status')!='error':return old
        dest.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,'-u',str(Path(__file__).with_name('preflight.py')),'--task',task,'--episode',str(ep),'--gpu',str(a.gpu),'--output',str(dest)]
        if screen:
            cmd+=['--screen-nearby']
            if a.candidate_file:cmd+=['--screen-candidates-file',str(a.candidate_file)]
        else:cmd+=['--anchor-local-offset',*[str(v) for v in offset],'--execute-transition','--execute-operation']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),MUJOCO_EGL_DEVICE_ID=str(a.gpu),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',
                 __EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/10_nvidia.json',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
                 PYTHONPATH='/data4/xuchenfei/robocasa365/third_party/robocasa:/data4/xuchenfei/robocasa365/third_party/robosuite')
        with (dest/'run.log').open('a') as f:
            proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=900)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                code=-1
        d=read_result(dest/'result.json')
        if code or d.get('status') not in terminal:
            d.update(status='error',error=f'worker exit={code}; prior status={d.get("status")}')
            (dest/'result.json').write_text(json.dumps(d,indent=2))
        return d
    def shortlist(candidates):
        if a.directional_shortlist:
            # Reserve slots for each yaw within every 5 mm score band.
            free=sorted((c for c in candidates if not c['collision']),key=lambda c:c['distance_score_m'])
            selected=[];counts={}
            for c in free:
                group=(int(c['distance_score_m']/.005),round(float(c['offset'][2]),4))
                if counts.get(group,0)>=2:continue
                selected.append(c);counts[group]=counts.get(group,0)+1
                if len(selected)>=a.max_attempts:break
            return selected
        if a.dense_shortlist:
            # More coverage per narrow distance band; all selected bands run near to far.
            free=sorted((c for c in candidates if not c['collision']),key=lambda c:c['distance_score_m'])
            selected=[];counts={}
            for c in free:
                band=int(c['distance_score_m']/.005)
                if counts.get(band,0)>=12:continue
                selected.append(c);counts[band]=counts.get(band,0)+1
                if len(selected)>=a.max_attempts:break
            return selected
        # Retain nearby choices first, while avoiding using the entire rollout
        # budget on almost identical headings. Distance shells expand outward.
        selected=[];used=set()
        for low,high in [(0,.011),(.011,.021),(.021,.031),(.031,.051),(.051,.076),(.076,.101),(.101,.16)]:
            pool=[c for c in candidates if not c['collision'] and low<=c['distance_score_m']<high]
            chosen=[]
            for c in pool:
                if len(chosen)>=4:break
                if any(np.linalg.norm(np.array(c['offset'][:2])-np.array(z['offset'][:2]))<.004 and abs(c['offset'][2]-z['offset'][2])<1 for z in chosen):continue
                chosen.append(c);used.add(c['candidate_id'])
            selected+=chosen
        return selected[:a.max_attempts]
    def run(job):
        task,ep=job;out=a.output/task/f'episode_{ep:06d}'
        with LOCK:progress[job]='screening';publish()
        result=run_preflight(task,ep,out/'screen',screen=True)
        if result.get('status')!='screen_completed':
            with LOCK:progress[job]='screen_error';publish()
            return
        candidates=json.loads((out/'screen/screen_candidates.json').read_text())
        if a.previous_output:
            previous=list(csv.DictReader((a.previous_output/'candidates.csv').open()))
            previous=[r for r in previous if r['task']==task and int(r['episode'])==ep]
            successful=[float(r['distance_score_m']) for r in previous if r['status']=='operation_success']
            bound=min(successful) if successful else float('inf')
            key=lambda v:tuple(round(float(x),8) for x in v)
            tested={key([r['local_dx_m'],r['local_dy_m'],r['delta_yaw_deg']]) for r in previous if r['status']!='error'}
            candidates=[c for c in candidates if c['distance_score_m']<bound-1e-8 and key(c['offset']) not in tested]
        options=shortlist(candidates)
        collected=[]
        for c in options:
            with LOCK:progress[job]='executing '+c['candidate_id'];publish()
            dest=out/c['candidate_id'];d=run_preflight(task,ep,dest,c['offset']);op=d.get('operation',{})
            success=d.get('status')=='operation_success' and op.get('success_ever') is True
            preview=''
            if success:
                views=[]
                for cam in ['robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand']:
                    cap=cv2.VideoCapture(str(dest/('transition_'+cam+'.mp4')));ok,im=cap.read();cap.release()
                    if ok:views.append(im)
                if len(views)==3:
                    preview=str(dest/'anchor_start.jpg');cv2.imwrite(preview,np.hstack(views))
            row={'task':task,'episode':ep,'candidate_id':c['candidate_id'],
                 'local_dx_m':c['offset'][0],'local_dy_m':c['offset'][1],'delta_yaw_deg':c['offset'][2],
                 'translation_m':c['translation_m'],'distance_score_m':c['distance_score_m'],
                 'status':d.get('status'),'success':success,'operation_steps':op.get('steps'),
                 'time_ratio':op.get('tracking',{}).get('time_ratio_to_consumed_source'),
                 'result_path':str(dest/'result.json'),'video':str(dest/'robot0_agentview_left.mp4') if success else '',
                 'preview':preview,'failure':d.get('error') or d.get('failure_reason') or d.get('transition',{}).get('failure_reason') or ('' if success else 'task_condition_not_reached')}
            collected.append(row)
            with LOCK:records[job]=collected;publish()
            if sum(x['success'] for x in collected)>=a.successes_per_episode:break
        with LOCK:progress[job]='done_success' if any(x['success'] for x in collected) else ('no_collision_free_candidate' if not options else 'no_success_within_budget');publish()
    with LOCK:publish()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for f in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):f.result()
    (a.output/'finished.txt').write_text(time.strftime('%F %T'))

if __name__=='__main__':main()
