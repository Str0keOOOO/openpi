"""Real-scene IK preflight for one Human episode at its CSV Anchor."""

from openpi.shared.paths import workspace_path
import argparse,csv,importlib.util,json,math,traceback,re,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp

def main(argv=None):
 def parse_range(value):
  try:
   start,end=(int(part) for part in value.split(':',1))
  except Exception as exc:
   raise argparse.ArgumentTypeError('Expected inclusive START:END, e.g. 218:232') from exc
  if start < 0 or end < start:
   raise argparse.ArgumentTypeError('Expected 0 <= START <= END')
  return (start,end)
 p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--episode',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--execute-transition',action='store_true');p.add_argument('--execute-operation',action='store_true');p.add_argument('--adaptive-interpolation',action='store_true',help='Locally split only saturated EE targets during operation replay');p.add_argument('--min-interpolation-substeps',type=int,default=2);p.add_argument('--max-interpolation-substeps',type=int,default=4);p.add_argument('--interpolate-source-range',action='append',type=parse_range,default=[],help='Also split each inclusive source frame range, e.g. 218:232');p.add_argument('--source-range-substeps',type=int,default=2);p.add_argument('--tracking-max-time-ratio',type=float,default=1.30);p.add_argument('--tracking-position-tolerance',type=float,default=.005);p.add_argument('--tracking-rotation-tolerance',type=float,default=.04);p.add_argument('--anchor-local-offset',type=float,nargs=3,default=[0.,0.,0.],metavar=('X_M','Y_M','YAW_DEG'));p.add_argument('--reference-anchor-extras',type=Path);p.add_argument('--restore-initial-extras',type=Path);p.add_argument('--delta11',type=float,nargs=11);p.add_argument('--handoff-rank',type=int,default=0);p.add_argument('--handoff-duration-scale',type=float,default=1.);p.add_argument('--minimum-predicted-path-saving-m',type=float,default=0.); p.add_argument('--screen-candidates-file',type=Path);p.add_argument('--screen-nearby',action='store_true');a=p.parse_args([format(float(v),'.15f') if re.fullmatch(r'-[0-9]+(?:\.[0-9]*)?[eE][+-]?[0-9]+',v) else v for v in (sys.argv[1:] if argv is None else argv)])
 a.output.mkdir(parents=True,exist_ok=True);result={'task':a.task,'episode':a.episode,'gpu':a.gpu,'status':'running','stage':'read'}
 def save(): (a.output/'result.json').write_text(json.dumps(result,indent=2))
 save();env=None
 try:
  # A failed physical GPU changes EGL enumeration independently of NVML.
  # Augmentation workers select a CUDA UUID; bind EGL to that same device.
  import os
  selected_uuid=os.environ.get('CUDA_VISIBLE_DEVICES','')
  if selected_uuid.startswith('GPU-') and ',' not in selected_uuid:
   sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'robocasa_augmentation'))
   from gpu_routing import egl_device_for_uuid
   egl_index=egl_device_for_uuid(selected_uuid)
   os.environ['MUJOCO_EGL_DEVICE_ID']=str(egl_index)
   # robosuite checks numeric EGL ordinals against CUDA_VISIBLE_DEVICES;
   # rendering device was already resolved unambiguously from its UUID.
   os.environ['CUDA_VISIBLE_DEVICES']=str(egl_index)
   result['rendering_gpu_mapping']={'physical_gpu':a.gpu,'uuid':selected_uuid,'egl_index':egl_index}
   save()
  import robocasa,robosuite,mujoco
  from robocasa.utils import lerobot_utils as LU
  from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
  from robosuite.utils.ik_utils import IKSolver
  root=workspace_path('data/train');data=root/a.task/'lerobot'
  row=None if a.reference_anchor_extras else next(r for r in csv.DictReader((root/'anchor_records.csv').open()) if r['task']==a.task and int(r['filtered_episode_index'])==a.episode)
  meta=LU.get_episode_meta(data,a.episode);states=LU.get_episode_states(data,a.episode);xml=LU.get_episode_model_xml(data,a.episode);em=LU.get_env_metadata(data)
  kw=dict(em['env_kwargs']);kw.update(env_name=em['env_name'],has_renderer=False,has_offscreen_renderer=True,use_camera_obs=False,camera_names=['robot0_agentview_left'],camera_widths=256,camera_heights=256,render_gpu_device_id=a.gpu,seed=0,ignore_done=True,randomize_cameras=False)
  env=robosuite.make(**kw);reset_to(env,dict(model=xml,states=states[0],ep_meta=json.dumps(meta)))
  robot=env.robots[0];site=robot.composite_controller.part_controllers['right'].ref_name;sid=env.sim.model.site_name2id(site)
  human_pos=np.array(env.sim.data.site_xpos[sid]);human_rot=np.array(env.sim.data.site_xmat[sid]).reshape(3,3).copy()
  source_gripper_qpos=np.asarray(env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes['right']]).copy()
  # Deterministic robot.reset() sets arm and torso qpos but deliberately does
  # not overwrite gripper qpos.  Set the official gripper initial state
  # explicitly and make the newly placed robot start at rest.
  robot.reset(deterministic=True)
  robot_velocity_ids=[env.sim.model.get_joint_qvel_addr(name) for name in robot.robot_model.all_joints]
  for arm in robot.arms:
   if robot.has_gripper[arm]:
    env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes[arm]]=robot.gripper[arm].init_qpos
    robot_velocity_ids.extend(robot._ref_gripper_joint_vel_indexes[arm])
  env.sim.data.qvel[robot_velocity_ids]=0.0
  env.sim.forward()
  from anchor_pose import set_base_pose
  if a.reference_anchor_extras:
   import gzip
   ax=gzip.open(a.reference_anchor_extras/'model.xml.gz','rt').read()
   if ax != xml:raise ValueError('Reference Anchor and Human scene XML differ')
   anchor_state=np.load(a.reference_anchor_extras/'states.npz')['states'][0]
   previous=env.sim.get_state().flatten().copy()
   env.sim.set_state_from_flattened(anchor_state);env.sim.forward()
   bid=env.sim.model.body_name2id('mobilebase0_base')
   nominal_xy=np.asarray(env.sim.data.body_xpos[bid])[:2].copy()
   nominal_yaw=Rotation.from_quat(np.asarray(env.sim.data.body_xquat[bid])[[1,2,3,0]]).as_euler('xyz')[2]
   anchor_qpos=env.sim.data.qpos.copy()
   env.sim.set_state_from_flattened(previous)
   delta=np.asarray(a.delta11,dtype=float)
   if delta.shape!=(11,) or not np.isfinite(delta).all():raise ValueError('Expected finite delta11')
   for name in robot.robot_model.all_joints:
    addr=env.sim.model.get_joint_qpos_addr(name);env.sim.data.qpos[addr]=anchor_qpos[addr]
   for arm in robot.arms:
    if robot.has_gripper[arm]:
     ids=robot._ref_gripper_joint_pos_indexes[arm];env.sim.data.qpos[ids]=anchor_qpos[ids]
   names=['mobilebase0_joint_torso_height']+[f'robot0_joint{i}' for i in range(1,8)]
   for name,change in zip(names,[delta[3],*np.radians(delta[4:])]):
    jid=env.sim.model.joint_name2id(name);addr=env.sim.model.get_joint_qpos_addr(name)
    requested=anchor_qpos[addr]+change
    if env.sim.model.jnt_limited[jid] and not env.sim.model.jnt_range[jid,0]<=requested<=env.sim.model.jnt_range[jid,1]:raise ValueError('Requested joint out of range: '+name)
    env.sim.data.qpos[addr]=requested
   env.sim.data.qvel[robot_velocity_ids]=0;env.sim.forward()
   a.anchor_local_offset=delta[:3].tolist()
   result['reference_anchor_extras']=str(a.reference_anchor_extras)
  else:
   nominal_yaw=math.radians(float(row['anchor_yaw_deg']));nominal_xy=np.array([float(row['anchor_x']),float(row['anchor_y'])])
  def candidate_pose(offset):
   dx,dy,dyaw=offset
   basis=np.array([[math.cos(nominal_yaw),-math.sin(nominal_yaw)],[math.sin(nominal_yaw),math.cos(nominal_yaw)]])
   return (nominal_xy+basis@np.array([dx,dy])).tolist(),nominal_yaw+math.radians(dyaw)
  xy,yaw=candidate_pose(a.anchor_local_offset)
  result['nominal_csv_anchor']={'xy_m':nominal_xy.tolist(),'yaw_deg':math.degrees(nominal_yaw)}
  result['anchor_candidate_adjustment']={'local_dx_m':a.anchor_local_offset[0],'local_dy_m':a.anchor_local_offset[1],'delta_yaw_deg':a.anchor_local_offset[2]}
  result['anchor_check']=set_base_pose(env,xy,yaw,np.array([0.,0.,yaw]))
  if a.restore_initial_extras:
   exact=np.load(a.restore_initial_extras/'states.npz')['states'][0]
   env.sim.set_state_from_flattened(exact);env.sim.forward()
   result['restored_exact_initial_extras']=str(a.restore_initial_extras)
  if a.screen_nearby:
   from nearby_candidates import offsets
   from robocasa.utils.env_utils import detect_robot_collision
   env.robot_geom_ids=None
   nominal_state=env.sim.get_state().flatten().copy()
   candidates=[]
   for item in (json.loads(a.screen_candidates_file.read_text()) if a.screen_candidates_file else offsets()):
    env.sim.set_state_from_flattened(nominal_state)
    cxy,cyaw=candidate_pose(item['offset'])
    set_base_pose(env,cxy,cyaw,np.array([0.,0.,cyaw]))
    collision=bool(detect_robot_collision(env))
    candidates.append(dict(item,xy_world_m=list(cxy),yaw_world_deg=math.degrees(cyaw),collision=collision))
   (a.output/'screen_candidates.json').write_text(json.dumps(candidates,indent=2))
   result.update(status='screen_completed',stage='done',screened=len(candidates),collision_free=sum(not c['collision'] for c in candidates))
   save();return
  env.sim.forward();start_pos=np.array(env.sim.data.site_xpos[sid]);start_rot=np.array(env.sim.data.site_xmat[sid]).reshape(3,3).copy()
  # Use RoboCasa's own start-pose collision predicate, the same one used by
  # set_robot_base() when it accepts/rejects randomized XY candidates. This is
  # intentionally a single-state check at the exact CSV Anchor; we must not
  # resample or move the user's specified Anchor. Mid-operation contacts are
  # recorded separately because intended grasp/manipulation contacts are valid.
  from robocasa.utils.env_utils import detect_robot_collision
  env.robot_geom_ids=None  # XML/model changed per episode; do not reuse cached IDs.
  official_collision=bool(detect_robot_collision(env))
  robot_geom_ids=set(env.robot_geom_ids or ())
  anchor_contacts=[]
  for ci in range(env.sim.data.ncon):
   contact=env.sim.data.contact[ci]
   g1,g2=int(contact.geom1),int(contact.geom2)
   if (g1 in robot_geom_ids)==(g2 in robot_geom_ids):
    continue
   pair=[]
   for gid in (g1,g2):
    body=int(env.sim.model.geom_bodyid[gid])
    pair.append({'geom_id':gid,'geom':env.sim.model.geom_id2name(gid),
                 'body':env.sim.model.body_id2name(body),
                 'is_robot_geom':gid in robot_geom_ids})
   anchor_contacts.append({'geoms':pair,'distance_m':float(contact.dist)})
  result['official_anchor_collision']={
   'collision':official_collision,
   'checker':'robocasa.utils.env_utils.detect_robot_collision',
   'semantics':'Any MuJoCo contact pair between a robot geom and a non-robot geom; no penetration-depth threshold and no target-contact exemption.',
   'contact_pairs':anchor_contacts,
  }
  if official_collision:
   result.update(status='anchor_initial_collision',stage='done',
    failure_reason='RoboCasa official detector found robot-to-environment contact at the requested Anchor pose (CSV plus recorded adjustment).')
   save()
   return
  arm_qpos=np.asarray(env.sim.data.qpos[robot._ref_joint_pos_indexes]).copy()
  torso_qpos=np.asarray(env.sim.data.qpos[robot._ref_torso_joint_pos_indexes]).copy()
  if not a.reference_anchor_extras and not np.allclose(arm_qpos,np.asarray(robot.init_qpos),rtol=0,atol=1e-10):
   raise RuntimeError('Anchor arm qpos does not match official nominal initial values')
  torso_expected=robot.init_torso_qpos if robot.init_torso_qpos is not None else robot.robot_model.init_torso_qpos
  if not a.reference_anchor_extras and torso_expected is not None and not np.allclose(torso_qpos,np.asarray(torso_expected),rtol=0,atol=1e-10):
   raise RuntimeError('Anchor torso qpos does not match official initial values')
  grippers={}
  for arm in robot.arms:
   if robot.has_gripper[arm]:
    actual=np.asarray(env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes[arm]]).copy()
    if not a.reference_anchor_extras and not np.allclose(actual,np.asarray(robot.gripper[arm].init_qpos),rtol=0,atol=1e-10):
     raise RuntimeError(f'Anchor {arm} gripper qpos does not match official initial values')
    grippers[arm]=actual.tolist()
  robot_qvel=np.asarray(env.sim.data.qvel[robot_velocity_ids]).copy()
  if np.any(robot_qvel != 0): raise RuntimeError('Anchor robot joint velocities are not zero')
  base_id=env.sim.model.body_name2id('mobilebase0_base')
  base_xyz=np.asarray(env.sim.data.body_xpos[base_id]).copy()
  base_wxyz=np.asarray(env.sim.data.body_xquat[base_id]).copy()
  base_rpy=Rotation.from_quat(base_wxyz[[1,2,3,0]]).as_euler('xyz')
  if abs(base_xyz[2]) > 1e-8 or np.max(np.abs(base_rpy[:2])) > 1e-8:
   raise RuntimeError(f'Non-default base height/tilt: z={base_xyz[2]}, roll/pitch={base_rpy[:2]}')
  result['initial_base_height_tilt_check']={'z_m':float(base_xyz[2]),'roll_rad':float(base_rpy[0]),'pitch_rad':float(base_rpy[1])}
  initial_sim_state=env.sim.get_state().flatten().copy()
  np.savez_compressed(a.output/'initial_state.npz',sim_state=initial_sim_state,
   arm_qpos=arm_qpos,torso_qpos=torso_qpos,robot_qvel=robot_qvel,
   base_world_xyz=np.asarray(env.sim.data.body_xpos[base_id]).copy(),
   base_world_quat_wxyz=np.asarray(env.sim.data.body_xquat[base_id]).copy(),
   ee_world_xyz=start_pos,ee_world_rotation=start_rot)
  result['initial_robot_state']={'arm_qpos_rad':arm_qpos.tolist(),'torso_qpos_m':torso_qpos.tolist(),
   'gripper_qpos':grippers,'robot_joint_velocities_zero':True,
   'base_world_xyz_m':np.asarray(env.sim.data.body_xpos[base_id]).tolist(),
   'base_world_quat_wxyz':np.asarray(env.sim.data.body_xquat[base_id]).tolist(),
   'ee_world_xyz_m':start_pos.tolist(),'initial_state_file':str(a.output/'initial_state.npz')}
  from smooth_join_plan import select_plans,curve
  source_actions=LU.get_episode_actions(data,a.episode)
  plans=select_plans(env,states,source_actions,start_pos,start_rot,duration_scale=a.handoff_duration_scale)
  (a.output/'handoff_candidates.json').write_text(json.dumps(plans,indent=2))
  if a.handoff_rank>=len(plans):raise ValueError('No such handoff candidate rank')
  plan=plans[a.handoff_rank];join_frame=int(plan['join_frame'])
  human_pos=np.asarray(plan['position']);human_rot=np.asarray(plan['rotation'])
  gids=np.asarray(robot._ref_gripper_joint_pos_indexes['right'],int)
  source_gripper_qpos=states[join_frame,1+gids].copy()
  result['handoff_plan']=plan;result['handoff_selected_source_frame']=join_frame;save()
  if a.minimum_predicted_path_saving_m>0 and a.restore_initial_extras:
   oldprov=json.loads((a.restore_initial_extras/'provenance.json').read_text());oldn=int(oldprov['transition_frames'])
   with np.load(a.restore_initial_extras/'ee_world.npz') as oldz:oldpath=oldz['ee_xyz_quat_xyzw'][:oldn,:3]
   baseline=float(np.linalg.norm(np.diff(oldpath,axis=0),axis=1).sum())+plan['human_initial_approach_path_m']
   proposed=plan['path_length_m']+plan['remaining_precontact_path_m']
   result['predicted_path_saving_m']=baseline-proposed
   if baseline-proposed<a.minimum_predicted_path_saving_m:
    result.update(status='no_predicted_improvement',stage='done',failure_reason='No meaningful free-space path saving before the first protected event');return
  model=env.sim.model._model;data_native=env.sim.data._data
  names=robot.robot_model.joints
  for name in names:
   jid=model.joint(name).id
   if int(model.jnt_qposadr[jid])!=jid or int(model.jnt_dofadr[jid])!=jid: raise RuntimeError(f'IK solver joint index mapping unsafe: {name}')
  solver=IKSolver(model,data_native,dict(joint_names=names,end_effector_sites=[site],nullspace_gains=[0.0]*len(names)),damping=.05,integration_dt=.05,max_dq=2.0,input_action_repr='absolute',input_rotation_repr='axis_angle',input_ref_frame='world')
  solver.q0=np.array([data_native.qpos[model.joint(n).id] for n in names])
  rots=Rotation.from_matrix(np.stack([start_rot,human_rot]));slerp=Slerp([0,1],rots)
  qpath=[]; errors=[];collision=[]
  result.update(stage='ik',anchor_xy=xy,anchor_yaw_deg=math.degrees(yaw),start_ee_xyz=start_pos.tolist(),human_ee_xyz=human_pos.tolist())
  save()
  for alpha in np.linspace(0,1,21):
   points,rotations=curve(start_pos,start_rot,human_pos,human_rot,np.asarray(plan['terminal_velocity_world_m_s']),np.asarray(plan['terminal_angular_velocity_world_rad_s']),plan['duration_s'],[alpha]);target=points[0];rot=rotations[0];goal=np.r_[target,Rotation.from_matrix(rot).as_rotvec()]
   for _ in range(80):
    q=solver.solve(goal);data_native.qpos[solver.dof_ids]=q;mujoco.mj_forward(model,data_native)
    poserr=np.linalg.norm(data_native.site(sid).xpos-target);rerr=np.linalg.norm(Rotation.from_matrix(rot@data_native.site(sid).xmat.reshape(3,3).T).as_rotvec())
    if poserr<.003 and rerr<.03: break
   if a.reference_anchor_extras:
    for joint in names:
     jid=model.joint(joint).id;value=data_native.qpos[model.jnt_qposadr[jid]]
     if model.jnt_limited[jid] and not model.jnt_range[jid,0]-.001<=value<=model.jnt_range[jid,1]+.001:raise RuntimeError('IK joint-limit violation: '+joint)
    for contact in data_native.contact:
     g1,g2=int(contact.geom1),int(contact.geom2)
     if (g1 in robot_geom_ids)==(g2 in robot_geom_ids):continue
     rid=g1 if g1 in robot_geom_ids else g2
     body=model.body(int(model.geom_bodyid[rid])).name or ''
     hand=any(part in body.lower() for part in ['gripper','finger'])
     if float(contact.dist)<(-.010 if hand else -.005):raise RuntimeError('IK illegal robot penetration: '+body)
   qpath.append(q.copy());errors.append([float(poserr),float(rerr)])
   collision.append([float(data_native.contact[i].dist) for i in range(data_native.ncon) if float(data_native.contact[i].dist)<-.005])
   if poserr>=.003 or rerr>=.03: raise RuntimeError(f'IK failed at fraction {alpha:.2f}: position={poserr:.4f} m orientation={rerr:.4f} rad')
  np.savez_compressed(a.output/'ik_preflight.npz',joint_waypoints=np.array(qpath),errors=np.array(errors),start_ee=start_pos,target_ee=human_pos)
  result.update(status='ik_preflight_completed',stage='done',max_position_error_m=max(e[0] for e in errors),max_orientation_error_rad=max(e[1] for e in errors),deep_contact_counts=[len(c) for c in collision],note='IK preflight completed; requested transition and operation stages are executed below.')
  if a.execute_transition:
   from transition_smooth import run_transition
   result.update(stage='transition');save()
   result['transition']=run_transition(env,initial_sim_state,start_pos,start_rot,human_pos,human_rot,a.output,source_gripper_qpos,human_states=states,plan=plan)
   result.update(status=result['transition']['status'],stage='done')
   if a.execute_operation and result['transition']['status']=='transition_executed':
    from operation import run_operation
    result.update(stage='operation');save()
    human_actions=source_actions[join_frame:]
    result['operation']=run_operation(env,states[join_frame:],human_actions,a.output,
      adaptive_interpolation=a.adaptive_interpolation,
      min_interpolation_substeps=a.min_interpolation_substeps,
      max_interpolation_substeps=a.max_interpolation_substeps,
      interpolate_source_ranges=a.interpolate_source_range,
      source_range_substeps=a.source_range_substeps,
      tracking_max_time_ratio=a.tracking_max_time_ratio,
      tracking_position_tolerance=a.tracking_position_tolerance,
      tracking_rotation_tolerance=a.tracking_rotation_tolerance)
    result['operation']['source_start_frame_index']=join_frame
    result['operation']['source_frame_index_semantics']='relative to the retained Human suffix; add source_start_frame_index'
    (a.output/'operation_result.json').write_text(json.dumps(result['operation'],indent=2))
    if a.reference_anchor_extras:
     with np.load(a.output/'operation_actual.npz') as executed:
      actual_states=executed['sim_state_before_action']
     with np.load(a.output/'transition_actual.npz') as bridge:
      actual_states=np.concatenate([bridge['sim_state_before_action'],actual_states],axis=0)
     from soft_joint_limits import check_soft_joint_limits
     limit_report=check_soft_joint_limits(model,robot.robot_model.all_joints,actual_states)
     limit_report['policy']['scope']='combined recorded transition and operation states'
     result['soft_joint_limit_report']=limit_report
     (a.output/'soft_joint_limit_report.json').write_text(json.dumps(limit_report,indent=2))
     if not limit_report['accepted']:
      raise RuntimeError('Executed joint excursion exceeds small/transient allowance: '+str(limit_report['rejected']))
     # Verify nonadjacent robot self-penetrations on the saved executed states.
     # Adjacent/same-body contacts are structural; intended finger contacts remain allowed.
     for snapshot in actual_states:
      data_native.qpos[:]=snapshot[1:1+model.nq]
      data_native.qvel[:]=snapshot[1+model.nq:1+model.nq+model.nv]
      mujoco.mj_forward(model,data_native)
      for contact in data_native.contact:
       g1,g2=int(contact.geom1),int(contact.geom2)
       if g1 not in robot_geom_ids or g2 not in robot_geom_ids or float(contact.dist)>=-.005:continue
       b1,b2=int(model.geom_bodyid[g1]),int(model.geom_bodyid[g2])
       if b1==b2 or model.body_parentid[b1]==b2 or model.body_parentid[b2]==b1 or model.body_parentid[b1]==model.body_parentid[b2]:continue
       raise RuntimeError('Executed nonadjacent robot self-penetration')
    accepted=result['operation']['success_ever']
    result.update(status='operation_success' if accepted else 'operation_failed',stage='done')
 except Exception:
  result.update(status='error',error=traceback.format_exc())
 finally:
  if env is not None: env.close()
  save()
if __name__=='__main__':main()
