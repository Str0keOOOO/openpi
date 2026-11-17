"""Deterministic PandaOmron planar Anchor pose restoration."""
import math
import numpy as np

def set_base_pose(env,xy,yaw,anchor_ori):
 # Solve the base joint values relative to the *restored current state* using
 # MuJoCo's own finite-difference kinematics. This avoids assuming the root
 # body basis, joint reference offsets, or a particular mobile-base axis order.
 model=env.sim.model; data=env.sim.data; sim=env.sim
 base_id=model.body_name2id('mobilebase0_base')
 fwd='mobilebase0_joint_mobile_forward'; side='mobilebase0_joint_mobile_side'
 xy_names=(fwd,side); xy_addrs=[model.get_joint_qpos_addr(n) for n in xy_names]
 yaw_name='mobilebase0_joint_mobile_yaw'; yaw_addr=model.get_joint_qpos_addr(yaw_name)
 def base_yaw():
  aw,ax,ay,az=np.asarray(data.body_xquat[base_id],dtype=float)
  return math.atan2(2*(aw*az+ax*ay),1-2*(ay*ay+az*az))
 eps=1e-4; sim.forward(); yaw0=base_yaw(); yawq=float(data.qpos[yaw_addr])
 data.qpos[yaw_addr]=yawq+eps; sim.forward()
 yaw_jac=((base_yaw()-yaw0+math.pi)%(2*math.pi)-math.pi)/eps
 data.qpos[yaw_addr]=yawq; sim.forward()
 if not np.isfinite(yaw_jac) or abs(yaw_jac)<0.5:
  raise RuntimeError(f'mobile-base yaw Jacobian invalid: {yaw_jac}')
 dyaw=(float(yaw)-yaw0+math.pi)%(2*math.pi)-math.pi
 data.qpos[yaw_addr]=yawq+dyaw/yaw_jac
 sim.forward()
 # Changing yaw can rotate the base body origin around the hinge anchor. Solve
 # XY only after the target yaw has been applied, then validate the final pose.
 p0=np.asarray(data.body_xpos[base_id],dtype=float).copy()
 q0=[float(data.qpos[a]) for a in xy_addrs]
 J=np.zeros((2,2),dtype=float)
 for j,addr in enumerate(xy_addrs):
  data.qpos[addr]=q0[j]+eps; sim.forward()
  J[:,j]=(np.asarray(data.body_xpos[base_id],dtype=float)[:2]-p0[:2])/eps
  data.qpos[addr]=q0[j]; sim.forward()
 delta_xy=np.asarray(xy,dtype=float)-p0[:2]
 delta_q,_,rank,_=np.linalg.lstsq(J,delta_xy,rcond=None)
 if rank<2 or np.linalg.norm(J@delta_q-delta_xy)>1e-5:
  raise RuntimeError(f'mobile-base XY Jacobian cannot represent requested pose: residual={np.linalg.norm(J@delta_q-delta_xy)}')
 for addr,value in zip(xy_addrs,np.asarray(q0)+delta_q): data.qpos[addr]=value
 sim.forward()
 # Fail closed if the requested transform was not actually realized.
 base_id=model.body_name2id('mobilebase0_base')
 actual_xyz=np.asarray(data.body_xpos[base_id],dtype=float)
 actual_q=np.asarray(data.body_xquat[base_id],dtype=float) # MuJoCo wxyz
 aw,ax,ay,az=actual_q/np.linalg.norm(actual_q)
 actual_yaw=math.atan2(2*(aw*az+ax*ay),1-2*(ay*ay+az*az))
 yaw_err=(actual_yaw-float(yaw)+math.pi)%(2*math.pi)-math.pi
 xy_err=float(np.linalg.norm(actual_xyz[:2]-np.asarray(xy,dtype=float)))
 if xy_err>1e-3 or abs(yaw_err)>math.radians(0.1):
  raise RuntimeError(f'Anchor pose did not restore: xy_error={xy_err:.6g} m, yaw_error={math.degrees(yaw_err):.6g} deg')
 return {'xy_error_m':xy_err,'yaw_error_deg':math.degrees(yaw_err)}
