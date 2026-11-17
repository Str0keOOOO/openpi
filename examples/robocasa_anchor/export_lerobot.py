"""Copy Human data and append verified executed Anchor episodes using LeRobot."""
import argparse,csv,fcntl,json,os,shutil,tempfile
from pathlib import Path
import cv2,numpy as np,pandas as pd
from robocasa.utils import lerobot_utils as LU

CAMERAS=('robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand')
TASKS=('PickPlaceSinkToCounter','OpenDrawer','OpenStandMixerHead','PickPlaceCounterToCabinet')

def initialize(source,output):
 if output.exists():
  if not (output/'anchor_dataset_manifest.json').exists(): raise RuntimeError(f'Output exists but is not an initialized Anchor dataset: {output}')
  return
 temp=output.with_name(output.name+'.building')
 if temp.exists(): raise RuntimeError(f'Incomplete initialization exists: {temp}')
 shutil.copytree(source,temp,symlinks=False)
 mapping=list(csv.DictReader((source/'episode_index_map.csv').open()))
 for row in mapping:
  task=row['task'];idx=int(row['filtered_episode_index']);extra=temp/task/'lerobot'/'extras'/f'episode_{idx:06d}'
  if not extra.is_dir(): raise RuntimeError(f'Missing Human episode extras: {extra}')
  info={'data_type':'human','source_task':task,'source_filtered_episode_index':idx,'source_original_episode_index':int(row['source_episode_index'])}
  (extra/'provenance.json').write_text(json.dumps(info,indent=2))
 (temp/'anchor_dataset_manifest.json').write_text(json.dumps({'version':2,'source_human_dataset':str(source),'policy':'Independent Human copy; Anchor episodes appended only after exact CSV Anchor passes robocasa.utils.env_utils.detect_robot_collision, transition tracking/scene checks pass, and the task success condition is reached. Manipulation contact-depth flags are diagnostic only.','created_by':'export_lerobot.py'},indent=2))
 temp.rename(output)


def append_in_place(source,output,task,source_episode,pilot,data_type="anchor",refresh_stats=True):
 from robocasa.utils.lerobot_utils import LerobotDatasetWrapper
 src=source/task/'lerobot';dst=output/task/'lerobot'
 result=json.loads((pilot/'result.json').read_text());tr=json.loads((pilot/'transition_result.json').read_text());op=json.loads((pilot/'operation_result.json').read_text())
 if result.get('status')!='operation_success' or tr.get('status')!='transition_executed' or not op.get('success_ever'):
  raise RuntimeError('Pilot did not pass transition and task-success checks')
 official=result.get('official_anchor_collision')
 if not isinstance(official,dict) or official.get('checker')!='robocasa.utils.env_utils.detect_robot_collision':
  raise RuntimeError('Missing RoboCasa official collision result for the exact Anchor start pose')
 if official.get('collision') is not False:
  raise RuntimeError('RoboCasa official detector rejected the exact Anchor start pose')
 old=sorted(dst.glob('extras/episode_*/provenance.json'))
 for f in old:
  z=json.loads(f.read_text())
  if z.get('data_type')==data_type and z.get('source_filtered_episode_index')==source_episode and z.get('source_task')==task and (data_type=='anchor' or z.get('source_pilot')==str(pilot)):
   print('already exported',task,source_episode,z['episode_index']);return
 transition=np.load(pilot/'transition_actual.npz');operation=np.load(pilot/'operation_actual.npz')
 obs=np.concatenate([transition['observation_state'],operation['observation_state']]);acts_env=np.concatenate([transition['actions_env_order'],operation['actions_env_order']]);states=np.concatenate([transition['sim_state_before_action'],operation['sim_state_before_action']]);ee=np.concatenate([transition['ee_world_xyz_quat_xyzw'],operation['ee_world_xyz_quat_xyzw']]);split=len(transition['actions_env_order']);n=len(acts_env)
 if any(len(x)!=n for x in (obs,states,ee)) or n==0 or not np.isfinite(obs).all() or not np.isfinite(acts_env).all(): raise ValueError('Action/state/EE arrays are invalid or misaligned')
 episodes=[json.loads(x) for x in (src/'meta/episodes.jsonl').read_text().splitlines()];episode=next(x for x in episodes if x['episode_index']==source_episode);lang=episode['tasks'][0]
 source_row=pd.read_parquet(next((src/'data').glob(f'*/episode_{source_episode:06d}.parquet'))).iloc[0]
 action=LU.reorder_hdf5_action(acts_env,LU.get_modality_dict(src));
 ds=LerobotDatasetWrapper(repo_id='local/robocasa-anchor',root=dst,download_videos=False)
 newidx=ds.meta.total_episodes
 paths=[];caps=[]
 for c in CAMERAS:
  pair=[pilot/('transition_'+c+'.mp4'),pilot/(c+'.mp4')]
  if not all(x.exists() for x in pair):raise FileNotFoundError(str(pair))
  caps.append([cv2.VideoCapture(str(x)) for x in pair])
  if not all(x.isOpened() for x in caps[-1]):raise IOError(f'Cannot open videos {pair}')
 try:
  for i in range(n):
   frame={};part=0 if i<split else 1
   for c,pair in zip(CAMERAS,caps):
    ok,bgr=pair[part].read()
    if not ok: raise RuntimeError(f'Video ended at frame {i}: {c}')
    frame['observation.images.'+c]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
   frame.update({'observation.state':np.asarray(obs[i],dtype=np.float64),'action':np.asarray(action[i],dtype=np.float64),
    'annotation.human.task_description':np.array([int(source_row['annotation.human.task_description'])],dtype=np.int64),
    'annotation.human.task_name':np.array([int(source_row['annotation.human.task_name'])],dtype=np.int64),
    'next.reward':np.array([float(i==n-1)],dtype=np.float32),'next.done':np.array([i==n-1],dtype=bool)})
   ds.add_frame(frame,task=lang)
  for c,pair in zip(CAMERAS,caps):
   for cap in pair:
    ok,_=cap.read()
    if ok: raise RuntimeError(f'Video has extra frames: {c}')
  ds.save_episode()
 finally:
  for pair in caps:
   for cap in pair:cap.release()
 extra=dst/'extras'/f'episode_{newidx:06d}';extra.mkdir(parents=True,exist_ok=True)
 shutil.copy2(src/'extras'/f'episode_{source_episode:06d}'/'model.xml.gz',extra/'model.xml.gz')
 m=LU.get_episode_meta(src,source_episode);m['data_type']=data_type;m['source_filtered_episode_index']=source_episode;m['source_task']=task
 anchor=result['initial_robot_state'];m['init_robot_base_pos']=anchor['base_world_xyz_m'];m['init_robot_base_ori']=[0.,0.,np.deg2rad(float(result['anchor_yaw_deg']))]
 (extra/'ep_meta.json').write_text(json.dumps(m,indent=2))
 np.savez_compressed(extra/'states.npz',states=states)
 np.savez_compressed(extra/'ee_world.npz',ee_xyz_quat_xyzw=ee,transition_frames=split,final_sim_state=operation['final_sim_state'])
 mapping=next(r for r in csv.DictReader((source/'episode_index_map.csv').open()) if r['task']==task and int(r['filtered_episode_index'])==source_episode)
 provenance={'data_type':data_type,'episode_index':newidx,'source_task':task,'source_filtered_episode_index':source_episode,'source_original_episode_index':int(mapping['source_episode_index']),'source_pilot':str(pilot),'transition_frames':split,'operation_frames':n-split,'success':True,'official_anchor_collision':official,'transition_scene_drift_diagnostic':tr.get('scene_drift',[]),'transition_contact_diagnostics':tr.get('deep_contacts',[]),'operation_contact_diagnostics':op.get('deep_contacts',[]),'operation_interpolation':op.get('interpolation'),'nominal_csv_anchor':result.get('nominal_csv_anchor'),'anchor_candidate_adjustment':result.get('anchor_candidate_adjustment'),'scene_motion_and_contact_depth_flags_are_diagnostic_only':True}
 (extra/'provenance.json').write_text(json.dumps(provenance,indent=2))
 # Recalculate project-specific q01/q99 stats from the complete copied dataset.
 if refresh_stats:
  stats=LU.calculate_dataset_statistics([str(p) for p in (dst/'data').glob('*/episode_*.parquet')]);(dst/'meta/stats.json').write_text(json.dumps(stats,indent=2))
 print('exported',task,'source',source_episode,'anchor',newidx,'frames',n,'transition',split,flush=True)


def rebuild_episode_map(source, output):
 fields=['task','filtered_episode_index','source_episode_index','data_type','source_filtered_episode_index']
 rows=[]
 for r in csv.DictReader((source/'episode_index_map.csv').open()):
  rows.append(dict(r,data_type='human',source_filtered_episode_index=r['filtered_episode_index']))
 for task in TASKS:
  for f in sorted((output/task/'lerobot'/'extras').glob('episode_*/provenance.json')):
   r=json.loads(f.read_text())
   if r.get('data_type')=='anchor':
    rows.append(dict(task=task,filtered_episode_index=r['episode_index'],source_episode_index=r['source_original_episode_index'],data_type='anchor',source_filtered_episode_index=r['source_filtered_episode_index']))
 tmp=output/'episode_index_map.csv.tmp'
 with tmp.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
 tmp.replace(output/'episode_index_map.csv')


def append(source,output,task,source_episode,pilot):
 # Copy-on-write transaction: the existing task dataset remains readable until
 # the complete new episode, stats and provenance have been saved and checked.
 lock_dir=output/'.locks';lock_dir.mkdir(exist_ok=True)
 with (lock_dir/'export.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  dst=output/task/'lerobot';backup=output/task/'lerobot.export_backup'
  if backup.exists() and not dst.exists(): backup.rename(dst)
  elif backup.exists() and dst.exists(): shutil.rmtree(backup)
  rebuild_episode_map(source,output)
  for f in (dst/'extras').glob('episode_*/provenance.json'):
   r=json.loads(f.read_text())
   if r.get('data_type')=='anchor' and r.get('source_task')==task and r.get('source_filtered_episode_index')==source_episode:
    print('already exported',task,source_episode,r['episode_index']);return
  stage=output/'.transactions'/task
  if stage.exists(): shutil.rmtree(stage)
  staged_dst=stage/task/'lerobot';staged_dst.parent.mkdir(parents=True,exist_ok=True)
  shutil.copytree(dst,staged_dst)
  append_in_place(source,stage,task,source_episode,pilot)
  info=json.loads((staged_dst/'meta/info.json').read_text());idx=int(info['total_episodes'])-1
  extra=staged_dst/'extras'/f'episode_{idx:06d}'
  if not (extra/'provenance.json').exists() or not (extra/'states.npz').exists():
   raise RuntimeError('Staged episode is missing provenance or simulator states')
  dst.rename(backup)
  try: staged_dst.rename(dst)
  except Exception:
   backup.rename(dst)
   raise
  rebuild_episode_map(source,output)
  shutil.rmtree(backup)
  shutil.rmtree(stage)


def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('/data4/xuchenfei/robocasa365/data/household_augmented'));p.add_argument('--output',type=Path,required=True);p.add_argument('--task',choices=TASKS,required=True);p.add_argument('--episode',type=int,required=True);p.add_argument('--pilot',type=Path,required=True);a=p.parse_args()
 initialize(a.source,a.output);append(a.source,a.output,a.task,a.episode,a.pilot)
if __name__=='__main__':main()
