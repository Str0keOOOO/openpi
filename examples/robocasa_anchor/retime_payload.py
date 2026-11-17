"""Build a synchronized, same-index replacement only after a verified rollout."""
import json,os
from pathlib import Path
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
import cv2,imageio.v2 as imageio
from scipy.spatial.transform import Rotation
from robocasa.utils import lerobot_utils as LU
from lerobot.datasets.compute_stats import get_feature_stats,sample_indices
CAMERAS=('robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand')
def write_json(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(d,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x));tmp.replace(p)
def build(dataset,task,idx,pilot,payload,*,smooth_handoff=False,continuous_handoff=False,accept_completed_rollout=False):
 dataset,pilot,payload=map(Path,(dataset,pilot,payload));ds=dataset/task/'lerobot';old=ds/'extras'/f'episode_{idx:06d}';prov=json.loads((old/'provenance.json').read_text());r=json.loads((pilot/'result.json').read_text());tr=r['transition'];op=r['operation']
 if r['status']!='operation_success' or not op['success_ever']:raise ValueError('Task not successful')
 quality_warnings=[]
 def quality_rejection(condition,message):
  if not condition:return
  if not accept_completed_rollout:raise ValueError(message)
  quality_warnings.append(message)
 position_tolerance=.005 if continuous_handoff else .003
 quality_rejection(tr['final_position_error_m']>position_tolerance or tr['final_rotation_error_rad']>.04,'Final EE error')
 if continuous_handoff and tr.get('speed_profile',{}).get('method')!='continuous_free_space_clock_with_source_OSC_feedforward':raise ValueError('Not a verified continuous tracker result')
 if continuous_handoff:
  with np.load(pilot/'transition_actual.npz') as tracked:tracking_errors=tracked['tracking_error_after_action_m_rad']
  quality_rejection(float(np.quantile(tracking_errors[:,0],.95))>.02 or float(np.quantile(tracking_errors[:,1],.95))>.1,'Continuous free-space P95 tracking error exceeds 2cm/0.1rad')
 quality_rejection(tr.get('collision_review_required') or op.get('collision_review_required'),'Illegal contact flagged')
 quality_rejection(not (smooth_handoff or continuous_handoff) and tr['steps']>prov['transition_frames'],'New transition slower than old')
 with np.load(pilot/'transition_actual.npz') as z:obs=z['observation_state'];act=z['actions_env_order'];states=z['sim_state_before_action'];ee=z['ee_world_xyz_quat_xyzw']
 with np.load(old/'states.npz') as z:old_first=z['states'][0]
 if len(states)==0 or not np.allclose(states[0],old_first,atol=1e-9,rtol=0):raise ValueError('Initial sim state changed')
 split=len(act)
 with np.load(pilot/'operation_actual.npz') as z:
  obs=np.concatenate([obs,z['observation_state']]);act=np.concatenate([act,z['actions_env_order']]);states=np.concatenate([states,z['sim_state_before_action']]);ee=np.concatenate([ee,z['ee_world_xyz_quat_xyzw']]);final=z['final_sim_state']
 n=len(act)
 if smooth_handoff or continuous_handoff:
  with np.load(old/'ee_world.npz') as z:old_ee=z['ee_xyz_quat_xyzw'][:,:3]
  path=lambda pts:float(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum())
  old_length,new_length=path(old_ee),path(ee[:,:3])
  quality_rejection(n>len(old_ee),'New complete execution slower than old')
  quality_rejection(old_length-new_length<max(.01,.02*old_length),'Insufficient measured EE path reduction')
  def direction(pts,join):
   x=pts[join-1]-pts[max(0,join-6)];y=pts[min(len(pts)-1,join+4)]-pts[join]
   return float(x@y/(np.linalg.norm(x)*np.linalg.norm(y))) if min(np.linalg.norm(x),np.linalg.norm(y))>=.003 else None
  before=direction(old_ee,prov['transition_frames'])
  after=direction(ee[:,:3],int(tr['virtual_join_step']) if continuous_handoff else split)
  actual_boundary=direction(ee[:,:3],split)
  quality_rejection(before is not None and after is not None and after<before-.1,'Measured handoff direction became worse')
  quality_rejection(continuous_handoff and after is not None and after<-.25,'Continuous virtual join still reverses')
  quality_rejection(continuous_handoff and actual_boundary is not None and actual_boundary<-.25,'Continuous controller boundary reverses')
  def pause(poses,join):
   low=max(0,join-10);high=min(len(poses)-1,join+10)
   speeds=np.linalg.norm(np.diff(poses[low:high+1,:3],axis=0),axis=1)*20
   rot=Rotation.from_quat(poses[low:high+1,3:]).as_matrix();angular=Rotation.from_matrix(rot[1:]@rot[:-1].transpose(0,2,1)).magnitude()*20
   still=(speeds<.01)&(angular<.05);d=np.diff(np.r_[False,still,False].astype(int));runs=np.flatnonzero(d==-1)-np.flatnonzero(d==1)
   return float(max(runs,default=0)/20)
  old_pause=new_pause=boundary_pause=None
  if continuous_handoff:
   with np.load(old/'ee_world.npz') as z:old_poses=z['ee_xyz_quat_xyzw']
   old_pause=pause(old_poses,prov['transition_frames']);new_pause=pause(ee,int(tr['virtual_join_step']));boundary_pause=pause(ee,split)
   quality_rejection(max(new_pause,boundary_pause)>max(.15,old_pause+.1),'Continuous handoff adds a measurable stop')
  handoff_quality=dict(old_frames=len(old_ee),new_frames=n,old_path_m=old_length,new_path_m=new_length,old_handoff_cosine=before,new_handoff_cosine=after,new_actual_boundary_cosine=actual_boundary,old_join_longest_pause_s=old_pause,new_virtual_join_longest_pause_s=new_pause,new_controller_boundary_longest_pause_s=boundary_pause)
 act=LU.reorder_hdf5_action(act,LU.get_modality_dict(ds))
 if not np.isfinite(act).all() or not np.isfinite(obs).all():raise ValueError('Nonfinite rollout')
 src=ds/'data/chunk-000'/f'episode_{idx:06d}.parquet';table=pq.read_table(src);values={}
 for name in table.column_names:
  v=table[name][0].as_py();values[name]=[v]*n
 values.update({'action':act.tolist(),'observation.state':obs.tolist(),'frame_index':np.arange(n),'episode_index':np.full(n,idx),'timestamp':np.arange(n,dtype=np.float32)/20,'index':np.arange(n)+int(table['index'][0].as_py())})
 for name in ['next.done','next.reward']:
  if name in values:
   sample=table[name][0].as_py();arr=np.zeros(n,dtype=bool if name=='next.done' else np.float32);arr[-1]=True if name=='next.done' else 1.;values[name]=arr if not isinstance(sample,list) else [[x] for x in arr]
 payload.mkdir(parents=True,exist_ok=True);pq.write_table(pa.Table.from_pydict(values,schema=table.schema),payload/'episode.parquet')
 stats={}
 for key,value in values.items():
  if key.startswith('observation.images.'):continue
  arr=np.asarray(value)
  if arr.dtype.kind not in 'OUS':stats[key]=get_feature_stats(arr,axis=0,keepdims=arr.ndim==1)
 sample_ids=set(sample_indices(n));
 for cam in CAMERAS:
  images=[];count=0
  writer=imageio.get_writer(str(payload/(cam+'.mp4')),fps=20,codec='libx264',pixelformat='yuv420p',macro_block_size=2,ffmpeg_params=['-threads','1','-preset','veryfast'])
  try:
   for part in ['transition_'+cam,cam]:
    cap=cv2.VideoCapture(str(pilot/(part+'.mp4')))
    if not cap.isOpened():raise ValueError('Unreadable video '+part)
    try:
     while True:
      ok,bgr=cap.read()
      if not ok:break
      rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);writer.append_data(rgb)
      if count in sample_ids:images.append(rgb.transpose(2,0,1))
      count+=1
    finally:cap.release()
  finally:writer.close()
  if count!=n:raise ValueError('Video/action frame mismatch')
  ft=get_feature_stats(np.stack(images),axis=(0,2,3),keepdims=True)
  stats['observation.images.'+cam]={k:v if k=='count' else np.squeeze(v/255.,axis=0) for k,v in ft.items()}
 np.savez_compressed(payload/'states.npz',states=states);np.savez_compressed(payload/'ee_world.npz',ee_xyz_quat_xyzw=ee,transition_frames=split,final_sim_state=final)
 prov.update(soft_joint_limit_report=r.get('soft_joint_limit_report'),transition_frames=split,operation_frames=n-split,transition_speed_profile=tr['speed_profile'],previous_transition_frames=prov['transition_frames'],retimed_from_episode_index=idx,source_retime_pilot=str(pilot),transition_contact_diagnostics=tr.get('deep_contacts',[]),operation_contact_diagnostics=op.get('deep_contacts',[]),operation_interpolation=op.get('interpolation'))
 if smooth_handoff or continuous_handoff:
  resume=int(tr['source_resume_frame']) if continuous_handoff else int(r['handoff_plan']['join_frame'])
  prov.update(transition_method='same_human_precontact_quintic_join',human_operation_start_frame=resume,handoff_plan=r['handoff_plan'],handoff_quality=handoff_quality)
  with np.load(pilot/'operation_actual.npz') as z:source_indices=np.r_[np.full(split,-1,dtype=int),z['source_frame_index']+resume]
  mapping=dict(source_human_frame_index=source_indices,transition_frames=split)
  if continuous_handoff:
   with np.load(pilot/'transition_actual.npz') as z:fractions=np.r_[z['source_frame_fraction'],source_indices[split:]]
   mapping.update(source_human_frame_fraction=fractions,source_human_frame_index=np.where(fractions<0,-1,np.floor(fractions)).astype(int))
   prov.update(transition_method='continuous_free_space_clock_with_source_OSC_feedforward',virtual_source_join_frame=int(r['handoff_plan']['join_frame']),virtual_join_execution_frame=int(tr['virtual_join_step']),moving_handoff_tolerance=tr.get('handoff_tolerance'),continuous_tracking_error_p95_m=float(np.quantile(tracking_errors[:,0],.95)),continuous_tracking_rotation_error_p95_rad=float(np.quantile(tracking_errors[:,1],.95)))
  if Path(r['handoff_plan']['human_dataset']).resolve()==ds.resolve() and int(r['handoff_plan']['human_episode'])==idx:
   reference_fractions=mapping.get('source_human_frame_fraction',mapping['source_human_frame_index']).copy()
   previous_map=old/'source_frame_mapping.npz'
   if previous_map.exists():
    with np.load(previous_map) as pm:old_mapping=np.asarray(pm['source_human_frame_fraction'] if 'source_human_frame_fraction' in pm else pm['source_human_frame_index'],float)
   else:
    old_mapping=np.arange(len(old_ee),dtype=float)-int(json.loads((old/'provenance.json').read_text())['transition_frames'])+int(json.loads((old/'provenance.json').read_text()).get('human_operation_start_frame',0))
    old_mapping[:int(json.loads((old/'provenance.json').read_text())['transition_frames'])]=-1
   mapped=np.full(len(reference_fractions),-1.,float)
   for ii,ff in enumerate(reference_fractions):
    if ff<0:continue
    lo=int(np.floor(ff));hi=min(lo+1,len(old_mapping)-1)
    if lo>=len(old_mapping):raise ValueError('Reference frame mapping out of bounds')
    if old_mapping[lo]>=0 and old_mapping[hi]>=0:mapped[ii]=old_mapping[lo]+(ff-lo)*(old_mapping[hi]-old_mapping[lo])
   mapping.update(source_reference_frame_fraction=reference_fractions,source_human_frame_fraction=mapped,source_human_frame_index=np.where(mapped<0,-1,np.floor(mapped)).astype(int))
   prov.update(optimization_reference='same_episode_previously_successful_execution',source_reference_resume_frame=resume,human_operation_start_frame=int(mapped[split]) if mapped[split]>=0 else None)
  np.savez_compressed(payload/'source_frame_mapping.npz',**mapping)
 if accept_completed_rollout:
  prov.update(acceptance_policy='user_requested_completed_successful_new_rollout',quality_warnings=quality_warnings,task_success_verified=True)
 write_json(payload/'provenance.json',prov);write_json(payload/'episode_stats.json',dict(episode_index=idx,stats=stats));write_json(payload/'ready.json',dict(task=task,episode=idx,frames=n,old_transition_frames=prov['previous_transition_frames'],new_transition_frames=split,speed_profile=tr['speed_profile'],final_position_error_m=tr['final_position_error_m'],final_rotation_error_rad=tr['final_rotation_error_rad']))
 ready=json.loads((payload/'ready.json').read_text())
 if smooth_handoff or continuous_handoff:
  ready.update(handoff_quality=handoff_quality,human_operation_start_frame=resume,continuous_handoff=continuous_handoff)
  write_json(payload/'ready.json',ready)
 if accept_completed_rollout:
  ready.update(acceptance_policy='user_requested_completed_successful_new_rollout',quality_warnings=quality_warnings)
  write_json(payload/'ready.json',ready)
 return ready
