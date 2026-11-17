"""Generate one frozen scene and its 73 legal-start slots, without policy actions."""
import argparse,gzip,json,os,sys,time,traceback,hashlib
from pathlib import Path
import numpy as np
from grouped import *

def make_env(task,pair,seed,gpu):
 import robocasa,robosuite
 from robosuite.controllers import load_composite_controller_config
 controller=load_composite_controller_config(controller=None,robot='PandaOmron')
 return robosuite.make(env_name=task,robots='PandaOmron',controller_configs=controller,layout_ids=[pair['layout_id']],style_ids=[pair['style_id']],obj_instance_split='pretrain',seed=seed,randomize_cameras=False,initialization_noise=None,robot_spawn_deviation_pos_x=0.,robot_spawn_deviation_pos_y=0.,robot_spawn_deviation_rot=0.,has_renderer=False,has_offscreen_renderer=True,use_camera_obs=False,use_object_obs=True,camera_names=list(CAMERAS),camera_heights=256,camera_widths=256,render_gpu_device_id=gpu,ignore_done=True)

def nominal_robot(env):
 from robocasa.utils.env_utils import set_robot_to_position
 robot=env.robots[0];sim=env.sim;robot.reset(deterministic=True)
 sim.data.qpos[sim.model.get_joint_qpos_addr('mobilebase0_joint_mobile_yaw')]=0.
 set_robot_to_position(env,np.asarray(env.init_robot_base_pos_anchor))
 sim.data.qpos[robot._ref_joint_pos_indexes]=np.asarray(robot.init_qpos)
 sim.data.qpos[robot._ref_torso_joint_pos_indexes]=np.asarray(robot.init_torso_qpos)
 for arm in robot.arms:
  if robot.has_gripper[arm]:sim.data.qpos[robot._ref_gripper_joint_pos_indexes[arm]]=robot.gripper[arm].init_qpos
 for name in robot.robot_model.all_joints:sim.data.qvel[sim.model.get_joint_qvel_addr(name)]=0.
 for arm in robot.arms:
  if robot.has_gripper[arm]:sim.data.qvel[robot._ref_gripper_joint_vel_indexes[arm]]=0.
 env.init_robot_base_pos=np.asarray(env.init_robot_base_pos_anchor).tolist();env.init_robot_base_ori=np.asarray(env.init_robot_base_ori_anchor).tolist();sim.forward()
 robot.base_pos=sim.data.get_body_xpos(robot.robot_model.root_body).copy();robot.base_ori=sim.data.get_body_xmat(robot.robot_model.root_body).copy();cc=robot.composite_controller;cc.update_state()
 for ctrl in cc.part_controllers.values():ctrl.update(force=True)
 cc.reset()

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--task',required=True);p.add_argument('--scene',type=int,required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args();config=json.loads(a.config.read_text());directory=a.output/'scenes'/a.task/f'scene_{a.scene:02d}';directory.mkdir(parents=True,exist_ok=True);env=None;began=time.time()
 try:
  # Resolve the CUDA UUID to EGL's ordinal, as in the existing generation pipeline.
  sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'robocasa_augmentation'))
  from gpu_routing import egl_device_for_uuid
  uuid=os.environ['CUDA_VISIBLE_DEVICES'];egl=egl_device_for_uuid(uuid);os.environ['MUJOCO_EGL_DEVICE_ID']=str(egl);os.environ['CUDA_VISIBLE_DEVICES']=str(egl)
  audit=json.loads((a.output/'training_scene_audit.json').read_text());pair=audit['preferred_pairs'][a.scene];anchor_dir=directory/'anchor';marker=anchor_dir/'anchor.json';anchor=None
  if marker.exists():
   anchor=json.loads(marker.read_text());env=make_env(a.task,anchor['scene'],anchor['scene']['seed'],egl)
   with np.load(anchor_dir/'snapshot.npz') as z:arrays={k:z[k].copy() for k in z.files}
   restore(env,anchor,arrays,reload_model=True)
  else:
   for attempt in range(config['max_anchor_attempts_per_scene']):
    seed=config['seed']+config['tasks'].index(a.task)*10000+a.scene*100+attempt
    try:
     env=make_env(a.task,pair,seed,egl);env.reset();nominal_robot(env)
     actual,legality=apply(env,np.zeros(11))
     if legality['status']!='ready':raise ValueError('Nominal official Anchor is illegal: '+legality['status'])
     meta=env.get_ep_meta()
     if (int(meta['layout_id']),int(meta['style_id']))!=(pair['layout_id'],pair['style_id']):raise ValueError('Requested held-out kitchen not respected')
     if [pair['layout_id'],pair['style_id']] in audit['training_pairs']:raise ValueError('Training/test kitchen combination overlap')
     anchor_dir.mkdir(parents=True,exist_ok=True);raw=env.sim.model.get_xml()
     with gzip.open(anchor_dir/'model.xml.gz','wt') as z:z.write(raw)
     task_state={}
     for name in ['drawer_side','robot_side','behavior','drawer_id','init_robot_base_pos','init_robot_base_ori','init_robot_base_pos_anchor','init_robot_base_ori_anchor']:
      if hasattr(env,name):task_state[name]=getattr(env,name)
     arrays=snapshot(env);np.savez_compressed(anchor_dir/'snapshot.npz',**arrays);np.savez_compressed(anchor_dir/'states.npz',states=arrays['state'][None]);dump(anchor_dir/'ep_meta.json',meta)
     anchor=dict(task=a.task,scene_index=a.scene,directory=str(anchor_dir),scene=dict(**pair,seed=seed,object_split='pretrain'),ep_meta=meta,instruction=meta['lang'],task_state=task_state,rng_state=env.rng.bit_generator.state,xml_sha256=hashlib.sha256(raw.encode()).hexdigest(),source='official_task_reset_anchor_with_nominal_robot',gripper_current_action={arm:env.robots[0].gripper[arm].current_action.copy() for arm in env.robots[0].arms if env.robots[0].has_gripper[arm]},initial_legality=legality,held_out_from_training=True,robot_joint_names=env.robots[0].robot_model.all_joints,reference_state_file=str(anchor_dir/'states.npz'))
     state,paths,camera=images_and_state(env,anchor_dir);anchor.update(observation_state=state.tolist(),images=paths,cameras=camera);dump(marker,anchor)
     # Check exact replay before creating any perturbed starts.
     restore(env,anchor,arrays,reload_model=True);s2,p2,c2=images_and_state(env,anchor_dir/'roundtrip')
     import imageio.v2 as imageio
     maes={cam:float(np.abs(imageio.imread(paths[cam]).astype(float)-imageio.imread(p2[cam]).astype(float)).mean()) for cam in CAMERAS}
     if not np.allclose(s2,state,atol=1e-7,rtol=0) or max(maes.values())>.5:raise RuntimeError('Anchor roundtrip state or RGB mismatch: '+str(maes))
     dump(anchor_dir/'roundtrip_check.json',dict(passed=True,state_max_error=float(np.max(np.abs(s2-state))),rgb_mean_absolute_error_255=maes));break
    except Exception as ex:
     with (directory/'anchor_attempts.jsonl').open('a') as f:f.write(json.dumps(dict(attempt=attempt,seed=seed,layout_id=pair['layout_id'],style_id=pair['style_id'],error=str(ex)))+'\n')
     # An incomplete Anchor must never masquerade as a successfully restored one.
     if marker.exists():marker.unlink()
     if env is not None:env.close();env=None
     anchor=None
   if anchor is None:raise RuntimeError('No legal, exactly restorable official Anchor in scene attempt budget')
  cases=[x for x in plan(config) if x['task']==a.task and x['scene_index']==a.scene];outcomes=[]
  for case in cases:
   dest=directory/'starts'/case['id'];done=dest/'configuration.json'
   if done.exists():outcomes.append(json.loads(done.read_text()));continue
   dest.mkdir(parents=True,exist_ok=True);accepted=False;reason=None;actual=None;tries=1 if case['family']=='baseline' else config['max_sampling_attempts_per_slot']
   for attempt in range(tries):
    restore(env,anchor,arrays);requested=propose(config,case,attempt);actual,legality=apply(env,requested)
    if not np.allclose(requested,actual,atol=1e-7,rtol=0):legality['status']='invalid_actual_delta'
    elif not validate_delta(config,case,actual):legality['status']='invalid_actual_stratum'
    event=dict(attempt=attempt,requested=requested.tolist(),actual=actual.tolist(),**legality)
    with (dest/'attempts.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
    if legality['status']!='ready':reason=legality['status'];continue
    snap=snapshot(env);np.savez_compressed(dest/'state.npz',**snap);state,paths,cameras=images_and_state(env,dest)
    record=dict(case=case,accepted=True,actual_delta=actual.tolist(),requested_delta=requested.tolist(),attempts=attempt+1,anchor=str(marker),scene=anchor['scene'],instruction=anchor['instruction'],state_file=str(dest/'state.npz'),observation_state=state.tolist(),images=paths,cameras=cameras,legality=legality,coordinate_definition='XY in actual Anchor base yaw-local frame; Yaw about world vertical; Height torso joint displacement; J1-J7 relative Anchor qpos',policy_actions_executed=0)
    dump(done,record);accepted=True;outcomes.append(record);break
   if not accepted:
    record=dict(case=case,accepted=False,attempts=tries,anchor=str(marker),scene=anchor['scene'],reason=reason,actual_delta=actual.tolist() if actual is not None else None,policy_actions_executed=0);dump(done,record);outcomes.append(record)
  summary=dict(task=a.task,scene_index=a.scene,status='completed',planned=len(cases),valid=sum(x['accepted'] for x in outcomes),invalid=sum(not x['accepted'] for x in outcomes),seconds=time.time()-began,scene=anchor['scene'],anchor=str(marker),policy_actions_executed=0);dump(directory/'summary.json',summary);print(json.dumps(summary),flush=True)
 except Exception:dump(directory/'summary.json',dict(task=a.task,scene_index=a.scene,status='error',error=traceback.format_exc(),seconds=time.time()-began));print(traceback.format_exc(),flush=True)
 finally:
  if env is not None:env.close()
if __name__=='__main__':main()
