"""Choose a pre-contact join within the same Human; no cross-scene demonstration mixing."""
import numpy as np,mujoco
from scipy.spatial.transform import Rotation

def skew(v):
 x,y,z=v;return np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
def left_jacobian(v):
 t=float(np.linalg.norm(v));k=skew(v)
 if t<1e-6:return np.eye(3)+.5*k+k@k/6
 return np.eye(3)+(1-np.cos(t))/t**2*k+(t-np.sin(t))/t**3*k@k

def curve(start_p,start_r,target_p,target_r,end_velocity,end_omega,duration,phase):
 s=np.asarray(phase);d=np.asarray(target_p)-start_p;v=np.asarray(end_velocity)*duration
 def poly(delta,terminal):return (10*delta-4*terminal)[None]*s[:,None]**3+(-15*delta+7*terminal)[None]*s[:,None]**4+(6*delta-3*terminal)[None]*s[:,None]**5
 p=np.asarray(start_p)[None]+poly(d,v)
 rv=Rotation.from_matrix(target_r@start_r.T).as_rotvec();wv=np.linalg.solve(left_jacobian(rv),end_omega)*duration
 r=Rotation.from_rotvec(poly(rv,wv)).as_matrix()@start_r
 return p,r

def select_plans(env,states,actions,start_p,start_r,max_frames=60,duration_scale=1.):
 sim=env.sim;model=sim.model._model;data=mujoco.MjData(model);robot=env.robots[0];ctrl=robot.composite_controller.part_controllers['right'];sid=sim.model.site_name2id(ctrl.ref_name);dt=1./env.control_freq
 hands={i for i in range(model.ngeom) if (model.body(int(model.geom_bodyid[i])).name or '').startswith('gripper0_')}
 robots={i for i in range(model.ngeom) if (model.body(int(model.geom_bodyid[i])).name or '').startswith(('robot0_','mobilebase0_','gripper0_'))}
 gids=np.asarray(robot._ref_gripper_joint_pos_indexes['right'],int);initial_grip=np.asarray(states[0])[1+gids]
 ps=[];rs=[];safe_end=min(max_frames,len(states)-6);boundary='maximum_precontact_search_window';contact_frame=None
 for t,state in enumerate(states[:min(max_frames+7,len(states))]):
  data.qpos[:]=state[1:1+model.nq];data.qvel[:]=state[1+model.nq:1+model.nq+model.nv];mujoco.mj_forward(model,data)
  ps.append(data.site_xpos[sid].copy());rs.append(data.site_xmat[sid].reshape(3,3).copy())
  if t>safe_end:continue
  contacted=any((int(c.geom1) in hands and int(c.geom2) not in robots or int(c.geom2) in hands and int(c.geom1) not in robots) and float(c.dist)<.002 for c in data.contact)
  grip_changed=np.max(np.abs(data.qpos[gids]-initial_grip))>.002
  # Free-space opening may be executed during the replacement approach.
  # Contact, closing, and all body commands must remain in the retained suffix.
  closing=bool(actions[t,6]>.1) if t<len(actions) else False
  body_command=bool(np.any(np.abs(actions[t,7:11])>1e-5) or actions[t,-1]>0) if t<len(actions) else False
  if contacted or closing or body_command:
   safe_end=max(0,t-2);contact_frame=t;boundary='first_gripper_contact_or_closing_or_body_command';break
 # Collect enough forward points to estimate the join tangent, using no env.step.
 last=min(len(states),max(safe_end+6,len(ps)))
 for state in states[len(ps):last]:
  data.qpos[:]=state[1:1+model.nq];data.qvel[:]=state[1+model.nq:1+model.nq+model.nv];mujoco.mj_forward(model,data);ps.append(data.site_xpos[sid].copy());rs.append(data.site_xmat[sid].reshape(3,3).copy())
 ps=np.asarray(ps);rs=np.asarray(rs);speed=np.linalg.norm(np.diff(ps,axis=0),axis=1)/dt;moving=speed[speed>.005];vtyp=float(np.clip(np.quantile(moving,.75) if len(moving) else .08,.04,.25))
 angular=Rotation.from_matrix(rs[1:]@rs[:-1].transpose(0,2,1)).magnitude()/dt;turning=angular[angular>.03];wtyp=float(np.clip(np.quantile(turning,.75) if len(turning) else .3,.2,1.5))
 remaining=np.r_[np.cumsum(np.linalg.norm(np.diff(ps[:safe_end+1],axis=0),axis=1)[::-1])[::-1],0.] if safe_end else np.array([0.])
 candidates=[]
 for k in range(safe_end+1):
  j=max(k,min(k+3,len(ps)-1,contact_frame-1 if contact_frame is not None else len(ps)-1));v=(ps[j]-ps[k])/max((j-k)*dt,dt);v*=min(1,vtyp/max(np.linalg.norm(v),1e-12));omega=Rotation.from_matrix(rs[j]@rs[k].T).as_rotvec()/max((j-k)*dt,dt);omega*=min(1,wtyp/max(np.linalg.norm(omega),1e-12))
  distance=float(np.linalg.norm(ps[k]-start_p));angle=float(Rotation.from_matrix(rs[k]@start_r.T).magnitude());duration=max(.15,duration_scale*distance/vtyp,duration_scale*angle/wtyp)
  p,r=curve(start_p,start_r,ps[k],rs[k],v,omega,duration,np.linspace(0,1,41));length=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum());rotation=float(Rotation.from_matrix(r[1:]@r[:-1].transpose(0,2,1)).magnitude().sum())
  if length>max(2*distance,distance+.04):continue
  cost=length+float(remaining[k])+.03*rotation
  candidates.append(dict(join_frame=k,duration_s=duration,source_linear_speed_m_s=vtyp,source_angular_speed_rad_s=wtyp,terminal_velocity_world_m_s=v.tolist(),terminal_angular_velocity_world_rad_s=omega.tolist(),position=ps[k].tolist(),rotation=rs[k].tolist(),path_length_m=length,remaining_precontact_path_m=float(remaining[k]),human_initial_approach_path_m=float(remaining[0]),cost=cost,first_protected_event_frame=contact_frame,last_safe_join_frame=safe_end,boundary_reason=boundary,approach_gripper_action=(-1. if k>0 and np.all(actions[:k,6]<=0) else 0.)))
 if not candidates:raise RuntimeError('No bounded smooth join candidate')
 candidates.sort(key=lambda x:x['cost']);return candidates
