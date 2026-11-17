"""Fixed grouped-perturbation benchmark: CPU plan, exact snapshots, no policy sampling."""
import collections,gzip,hashlib,json,math
from pathlib import Path
import numpy as np

CAMERAS=('robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand')
STATE_KEYS=('robot0_base_pos','robot0_base_quat','robot0_base_to_eef_pos','robot0_base_to_eef_quat','robot0_gripper_qpos')
MODEL_ARRAYS=('body_pos','body_quat','cam_pos','cam_quat','cam_fovy')
FAMILIES={'body':list(range(4)),'arm':list(range(4,11)),'joint':list(range(11))}

def dump(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(d,indent=2,default=lambda x:np.asarray(x).tolist()));q.replace(p)

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def rng_for(*items):return np.random.default_rng(int(digest(items)[:16],16))

def plan(config):
 rows=[];nscene=config['scenes_per_task'];nsample=config['samples_per_scene_level'];nbin=config['magnitude_subbins'];assert (nscene,nsample,nbin)==(5,8,5)
 for task in config['tasks']:
  for scene in range(nscene):rows.append(dict(task=task,scene_index=scene,id='baseline',family='baseline',level='baseline',sample_index=0,active_dimensions=[],signs=[0]*11,magnitude_subbins=[-1]*11,inference_seed=config['seed']+scene))
  for family,active in FAMILIES.items():
   for level in ['small','medium','large']:
    signs=np.zeros((nscene,nsample,11),int);bins=np.full((nscene,nsample,11),-1,int)
    for axis in active:
     rng=rng_for(config['seed'],task,family,level,axis);sp=rng.permutation(nscene);xp=rng.permutation(nsample);bp=rng.permutation(nbin);flip=int(rng.choice([-1,1]))
     for scene in range(nscene):
      for sample in range(nsample):
       bins[scene,sample,axis]=bp[(sp[scene]+xp[sample])%nbin]
       signs[scene,sample,axis]=1 if axis==3 else flip*(1 if xp[sample]<nsample//2 else -1)
    for scene in range(nscene):
     for sample in range(nsample):rows.append(dict(task=task,scene_index=scene,id=f'{family}_{level}_{sample:02d}',family=family,level=level,sample_index=sample,active_dimensions=active,signs=signs[scene,sample].tolist(),magnitude_subbins=bins[scene,sample].tolist(),inference_seed=config['seed']+scene*100+sample))
 assert len(rows)==1460
 return rows

def propose(config,case,attempt):
 delta=np.zeros(11)
 if case['family']=='baseline':return delta
 tier=['small','medium','large'].index(case['level']);rng=rng_for(config['seed'],case['task'],case['scene_index'],case['id'],attempt)
 for axis in case['active_dimensions']:
  lo,hi=config['level_multipliers'][tier];bound=config['training_limits'][axis];width=(hi-lo)*bound/config['magnitude_subbins'];low=lo*bound+case['magnitude_subbins'][axis]*width;high=low+width;low=max(low,np.nextafter(config['tolerances'][axis],np.inf))
  delta[axis]=case['signs'][axis]*float(rng.uniform(low,high))
 return delta

def validate_delta(config,case,actual):
 actual=np.asarray(actual);tol=np.asarray(config['tolerances']);active=np.flatnonzero(np.abs(actual)>tol).tolist()
 if not np.isfinite(actual).all() or active!=case['active_dimensions'] or actual[3]<0:return False
 if case['family']=='baseline':return True
 tier=['small','medium','large'].index(case['level']);lo,hi=config['level_multipliers'][tier]
 for axis in active:
  if np.sign(actual[axis])!=case['signs'][axis]:return False
  bound=config['training_limits'][axis];width=(hi-lo)*bound/config['magnitude_subbins'];low=lo*bound+case['magnitude_subbins'][axis]*width;high=low+width
  if not low-1e-7<abs(actual[axis])<=high+1e-7:return False
 return True

def train_scene_audit(root):
 pairs=collections.Counter();layouts=collections.Counter();styles=collections.Counter();episodes=0
 for p in Path(root).glob('*/lerobot/extras/episode_*/provenance.json'):
  prov=json.loads(p.read_text())
  if prov.get('data_type')!='human':continue
  m=json.loads(p.with_name('ep_meta.json').read_text());l,s=int(m['layout_id']),int(m['style_id']);pairs[(l,s)]+=1;layouts[l]+=1;styles[s]+=1;episodes+=1
 if episodes!=418:raise ValueError(f'Expected 418 actual source Humans, found {episodes}')
 options=[]
 for l in sorted(layouts,key=lambda k:(-layouts[k],k)):
  for s in sorted(styles,key=lambda k:(-styles[k],k)):
   if (l,s) not in pairs:options.append(dict(layout_id=l,style_id=s,layout_training_count=layouts[l],style_training_count=styles[s]))
 chosen=[];used_l=set();used_s=set()
 for row in options:
  if row['layout_id'] in used_l or row['style_id'] in used_s:continue
  chosen.append(row);used_l.add(row['layout_id']);used_s.add(row['style_id'])
  if len(chosen)==5:break
 if len(chosen)!=5:raise ValueError('Insufficient held-out layout/style combinations')
 return dict(human_episodes=episodes,training_pairs=[list(p) for p in sorted(pairs)],unique_training_pairs=len(pairs),preferred_pairs=chosen,candidate_pairs=options,rule='Frequency-ranked familiar layouts/styles with combinations absent from all 418 source Humans; no task-success selection; pretrain object split')

def snapshot(env):
 return dict(state=env.sim.get_state().flatten().copy(),qpos=env.sim.data.qpos.copy(),qvel=env.sim.data.qvel.copy(),ctrl=env.sim.data.ctrl.copy(),**{k:np.asarray(getattr(env.sim.model,k)).copy() for k in MODEL_ARRAYS})

def restore(env,anchor,arrays,*,reload_model=False):
 if reload_model:
  from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
  reset_to(env,dict(model=gzip.open(Path(anchor['directory'])/'model.xml.gz','rt').read(),ep_meta=json.dumps(anchor['ep_meta']),states=arrays['state']))
 for key in MODEL_ARRAYS:getattr(env.sim.model,key)[:]=arrays[key]
 env.sim.set_state_from_flattened(arrays['state']);env.sim.data.ctrl[:]=arrays['ctrl'];env.sim.forward()
 for name,value in anchor.get('task_state',{}).items():setattr(env,name,np.asarray(value) if name.startswith('init_robot_') else value)
 env.cur_time=0;env.timestep=0;env.done=False;env._obs_cache={};env._reset_observables()
 if 'rng_state' in anchor:env.rng.bit_generator.state=anchor['rng_state']
 before=env.sim.data.qpos.copy()
 for robot in env.robots:
  robot.base_pos=env.sim.data.get_body_xpos(robot.robot_model.root_body).copy();robot.base_ori=env.sim.data.get_body_xmat(robot.robot_model.root_body).copy();cc=robot.composite_controller;cc.update_state()
  for ctrl in cc.part_controllers.values():ctrl.update(force=True)
  cc.reset()
  for arm,value in anchor.get('gripper_current_action',{}).items():
   if arm in robot.gripper:robot.gripper[arm].current_action=np.asarray(value).copy()
 if not np.array_equal(before,env.sim.data.qpos):raise RuntimeError('Controller restore modified physical qpos')
 if reload_model:
  current=env.get_ep_meta()
  if current['lang']!=anchor['instruction'] or current['fixture_refs']!=anchor['ep_meta']['fixture_refs']:raise RuntimeError('Restored task language or fixture target differs')
 return env._get_observations(force_update=True)

def apply(env,requested):
 from scipy.spatial.transform import Rotation
 import sys
 sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'robocasa_anchor'))
 from anchor_pose import set_base_pose
 from robocasa.utils.env_utils import detect_robot_collision
 sim=env.sim;m=sim.model;d=sim.data;robot=env.robots[0];before=d.qpos.copy();bid=m.body_name2id('mobilebase0_base');base=d.body_xpos[bid].copy();yaw=Rotation.from_quat(np.asarray(d.body_xquat[bid])[[1,2,3,0]]).as_euler('xyz')[2];basis=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
 if np.any(requested[:3]):set_base_pose(env,base[:2]+basis@requested[:2],yaw+math.radians(requested[2]),None)
 names=['mobilebase0_joint_torso_height']+[f'robot0_joint{i}' for i in range(1,8)];addresses=[m.get_joint_qpos_addr(n) for n in names];d.qpos[addresses]+=np.r_[requested[3],np.radians(requested[4:])];sim.forward();limits=[]
 for name,address in zip(names,addresses):
  jid=m.joint_name2id(name)
  if m.jnt_limited[jid] and not m.jnt_range[jid,0]<=d.qpos[address]<=m.jnt_range[jid,1]:limits.append(dict(joint=name,value=float(d.qpos[address]),range=m.jnt_range[jid].tolist()))
 allowed={m.get_joint_qpos_addr(n) for n in robot.robot_model.all_joints};changed=set(np.flatnonzero(d.qpos!=before))
 if not changed<=allowed:raise RuntimeError('Non-robot qpos changed')
 actual_yaw=Rotation.from_quat(np.asarray(d.body_xquat[bid])[[1,2,3,0]]).as_euler('xyz')[2];actual=np.r_[basis.T@(d.body_xpos[bid][:2]-base[:2]),math.degrees((actual_yaw-yaw+math.pi)%(2*math.pi)-math.pi),d.qpos[addresses[0]]-before[addresses[0]],np.degrees(d.qpos[addresses[1:]]-before[addresses[1:]])]
 env.robot_geom_ids=None;collision=bool(detect_robot_collision(env));contact=[]
 if collision:
  robotids=set(env.robot_geom_ids)
  for c in d.contact:
   a,b=int(c.geom1),int(c.geom2)
   if (a in robotids)==(b in robotids):continue
   contact.append(dict(geoms=[m.geom_id2name(a),m.geom_id2name(b)],distance_m=float(c.dist)))
 return actual,dict(status='invalid_joint_limit' if limits else 'invalid_official_collision' if collision else 'ready',joint_limit_violations=limits,official_collision=collision,contacts=contact,checker='robocasa.utils.env_utils.detect_robot_collision')

def images_and_state(env,directory):
 import imageio.v2 as imageio
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);ob=env._get_observations(force_update=True);state=np.concatenate([np.asarray(ob[k]) for k in STATE_KEYS]);paths={};camera_poses={}
 if state.shape!=(16,) or not np.isfinite(state).all():raise RuntimeError('Invalid real 16D robot observation')
 for camera in CAMERAS:
  image=np.asarray(env.sim.render(width=256,height=256,camera_name=camera))[::-1].copy();path=directory/(camera+'.png');imageio.imwrite(path,image);paths[camera]=str(path)
  cid=env.sim.model.camera_name2id(camera);T=np.eye(4);T[:3,:3]=env.sim.data.cam_xmat[cid].reshape(3,3);T[:3,3]=env.sim.data.cam_xpos[cid];camera_poses[camera]=dict(T_world_camera_opengl=T.tolist(),fovy_deg=float(env.sim.model.cam_fovy[cid]),image_size=[256,256])
 return state,paths,camera_poses
