"""Continuous, smoothly time-scaled free-space tracking across the virtual join."""
from pathlib import Path
import json
import imageio.v2 as imageio
import mujoco,numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation,RotationSpline
from smooth_join_plan import curve
from transition_smooth import CAMERAS


def make_reference(env,states,plan,start_pos,start_rot):
    model=env.sim.model._model;data=mujoco.MjData(model);robot=env.robots[0];ctrl=robot.composite_controller.part_controllers['right'];sid=env.sim.model.site_name2id(ctrl.ref_name)
    k=int(plan['join_frame']);end=int(plan['last_safe_join_frame']);dt=1./env.control_freq;duration=float(plan['duration_s']);poses=[];rots=[];origins=[]
    oid=env.sim.model.site_name2id(f'{ctrl.naming_prefix}{ctrl.part_name}_center')
    for s in states[:end+5]:
        data.time=s[0];data.qpos[:]=s[1:1+model.nq];data.qvel[:]=s[1+model.nq:];mujoco.mj_forward(model,data)
        poses.append(data.site_xpos[sid].copy());rots.append(data.site_xmat[sid].reshape(3,3).copy());origins.append(data.site_xmat[oid].reshape(3,3).copy())
    poses=np.asarray(poses);rots=np.asarray(rots);origins=np.asarray(origins)
    t0=np.linspace(0,duration,max(8,int(np.ceil(duration/dt))*2+1));p0,r0=curve(start_pos,start_rot,poses[k],rots[k],np.asarray(plan['terminal_velocity_world_m_s']),np.asarray(plan['terminal_angular_velocity_world_rad_s']),duration,t0/duration)
    # Smooth vertical avoidance arc: zero displacement/velocity/acceleration at both ends.
    u=t0/duration;p0[:,2]+=float(plan.get('bridge_lift_m',0.))*64*u**3*(1-u)**3
    times=np.r_[t0,duration+np.arange(1,end-k+1)*dt];p=np.concatenate([p0,poses[k+1:end+1]]);r=np.concatenate([r0,rots[k+1:end+1]])
    last_v=(poses[min(end+1,len(poses)-1)]-poses[end])/dt if end+1<len(poses) else np.zeros(3)
    position=CubicSpline(times,p,bc_type=((1,np.zeros(3)),(1,last_v)));rotation=RotationSpline(times,Rotation.from_matrix(r))
    return dict(position=position,rotation=rotation,total=float(times[-1]),duration=duration,join=k,end=end,dt=dt,positions=poses,rotations=rots,origins=origins)


def run_transition(env,initial_sim_state,start_pos,start_rot,human_pos,human_rot,output,source_gripper_qpos,human_states=None,plan=None):
    if plan is None or human_states is None:raise ValueError('Real source trajectory and selected plan are required')
    from robocasa.utils import lerobot_utils as LU
    output=Path(output);sim=env.sim;robot=env.robots[0];cc=robot.composite_controller
    sim.set_state_from_flattened(initial_sim_state);sim.forward();cc.update_state()
    for c in cc.part_controllers.values():c.update(force=True)
    cc.reset();ctrl=cc.part_controllers['right'];sid=sim.model.site_name2id(ctrl.ref_name)
    if ctrl.input_type!='delta' or ctrl.input_ref_frame!='base':raise RuntimeError('Unsupported OSC representation')
    actions_source=LU.get_episode_actions(Path(plan['human_dataset']),int(plan['human_episode']))
    ref=make_reference(env,human_states,plan,start_pos,start_rot);dt=ref['dt'];duration=ref['duration'];end=ref['end'];k=ref['join']
    scale=(np.asarray(ctrl.output_max)-np.asarray(ctrl.output_min))/(np.asarray(ctrl.input_max)-np.asarray(ctrl.input_min));midin=(np.asarray(ctrl.input_max)+np.asarray(ctrl.input_min))/2;midout=(np.asarray(ctrl.output_max)+np.asarray(ctrl.output_min))/2
    source_delta=(np.clip(actions_source[:end+1,:6],ctrl.input_min,ctrl.input_max)-midin)*scale+midout
    offset_p=np.einsum('nij,nj->ni',ref['origins'][:len(source_delta)],source_delta[:,:3]);offset_w=np.einsum('nij,nj->ni',ref['origins'][:len(source_delta)],source_delta[:,3:])
    def gain(offset,step):
        den=np.sum(step*step,axis=1);good=(den>2.5e-7)&(np.sum(offset*step,axis=1)>0)
        ratios=np.sum(offset[good]*step[good],axis=1)/den[good]
        return float(np.clip(np.median(ratios),.5,6.)) if len(ratios) else 1.
    gp=gain(offset_p[:end],np.diff(ref['positions'][:end+1],axis=0));gw=gain(offset_w[:end],Rotation.from_matrix(ref['rotations'][1:end+1]@ref['rotations'][:end].transpose(0,2,1)).as_rotvec())
    split=cc._action_split_indexes;keys=('robot0_base_pos','robot0_base_quat','robot0_base_to_eef_pos','robot0_base_to_eef_quat','robot0_gripper_qpos')
    writers=[imageio.get_writer(str(output/('transition_'+cam+'.mp4')),fps=float(env.control_freq),codec='libx264',pixelformat='yuv420p',macro_block_size=2,ffmpeg_params=['-threads','1','-preset','veryfast']) for cam in CAMERAS]
    obs=[];states=[];actions=[];ees=[];phases=[];rates=[];errors=[];targets=[];contacts=[];source_ids=[];virtual_join_step=None;phase=0.;rate=1.;failed=None;max_steps=int(np.ceil(ref['total']/dt*1.6))+20
    posture=None
    if plan.get('source_posture_guidance'):
        from source_posture import SourcePosture
        posture=SourcePosture(ctrl,sim.model._model,robot.robot_model.joints,limit_margin_rad=float(plan.get('posture_limit_margin_rad',.035)))
    integral_p=np.zeros(3);integral_w=np.zeros(3);catchup_used=0;catchup_budget=int(plan.get('handoff_catchup_steps',0))
    try:
        for _ in range(max_steps):
            sim.forward();p=sim.data.site_xpos[sid].copy();r=sim.data.site_xmat[sid].reshape(3,3).copy();pr=ref['position'](phase);rr=ref['rotation'](phase).as_matrix();v=ref['position'](phase,1);omega=rr@ref['rotation'](phase,1)
            # A continuous clock: no zero-rate freeze and no endpoint settling.
            lag=float((pr-p)@v/max(np.linalg.norm(v),1e-9));cross=float(np.linalg.norm((pr-p)-lag*v/max(np.linalg.norm(v),1e-9)));angle=float(Rotation.from_matrix(rr@r.T).magnitude())
            desired=float(np.clip(1.-max(lag-.005,0)/.035-max(cross-.01,0)/.05-max(angle-.06,0)/.5,.4,1.));rate+=float(np.clip(desired-rate,-.06,.06));next_phase=min(ref['total'],phase+dt*rate)
            control_phase=phase if plan.get('reference_phase_aligned',False) else next_phase
            target_p=ref['position'](control_phase);target_r=ref['rotation'](control_phase).as_matrix();vel=ref['position'](control_phase,1)*rate;ang=target_r@ref['rotation'](control_phase,1)*rate
            src=float(np.clip(k+(control_phase-duration)/dt,0,end));i=int(np.floor(src));j=min(i+1,end);weight=src-i
            source_p=(1-weight)*offset_p[i]+weight*offset_p[j];source_w=(1-weight)*offset_w[i]+weight*offset_w[j]
            blend=float(np.clip((control_phase-duration+.2)/.4,0,1));blend=blend*blend*(3-2*blend)
            ff_p=(1-blend)*gp*vel*dt+blend*source_p*rate;ff_w=(1-blend)*gw*ang*dt+blend*source_w*rate
            # In the retained Human free approach this becomes exactly the
            # source OSC goal plus measured pose feedback, without a reset.
            gain_feedback=float(plan.get('continuous_feedback_gain',1.))
            ep=target_p-p;ew=Rotation.from_matrix(target_r@r.T).as_rotvec()
            integral_p=np.clip(integral_p+ep*dt,-.01,.01);integral_w=np.clip(integral_w+ew*dt,-.05,.05)
            ki=float(plan.get('continuous_integral_gain',0.))
            if phase>=ref['total']-1e-9:
                ff_p=np.zeros(3);ff_w=np.zeros(3);catchup_used+=1
            delta_world_p=gain_feedback*ep+ff_p+ki*integral_p;delta_world_w=gain_feedback*ew+ff_w+ki*integral_w
            _,origin=cc.get_controller_base_pose('right');origin=np.asarray(origin).reshape(3,3);physical=np.r_[origin.T@delta_world_p,origin.T@delta_world_w];normal=(physical-midout)/scale+midin;ratio=float(max(1.,np.max(np.abs(normal))))
            action=np.zeros(env.action_dim);a,b=split['right'];action[a:b]=normal/ratio;a,b=split['right_gripper'];action[a:b]=float(plan.get('approach_gripper_action',0.));action[-1]=-1.
            ob=env._get_observations(force_update=True);obs.append(np.concatenate([np.asarray(ob[key]) for key in keys]));states.append(sim.get_state().flatten().copy());ees.append(np.r_[p,Rotation.from_matrix(r).as_quat()]);targets.append(np.r_[ref['position'](next_phase),ref['rotation'](next_phase).as_quat()]);phases.append(next_phase);rates.append(rate);source_ids.append(src if next_phase>=duration else -1.)
            if virtual_join_step is None and next_phase>=duration:virtual_join_step=len(actions)
            for writer,cam in zip(writers,CAMERAS):writer.append_data(np.asarray(sim.render(width=256,height=256,camera_name=cam))[::-1])
            if posture is not None:
                reference_index=int(np.clip(round(k+(next_phase-duration)/dt),0,end))
                blend=min(1.,max(0.,next_phase/max(duration,dt)))
                posture.update(human_states[reference_index],blend)
            env.step(action);actions.append(action);sim.forward();actual_p=sim.data.site_xpos[sid];actual_r=sim.data.site_xmat[sid].reshape(3,3);errors.append([float(np.linalg.norm(actual_p-ref['position'](next_phase))),float(Rotation.from_matrix(ref['rotation'](next_phase).as_matrix()@actual_r.T).magnitude())])
            for c in sim.data.contact:
                if float(c.dist)>=-.005:continue
                geoms=[]
                for gid in [int(c.geom1),int(c.geom2)]:
                    bid=int(sim.model.geom_bodyid[gid]);geoms.append(dict(geom_id=gid,body=sim.model.body_id2name(bid)))
                names=[x['body']or'' for x in geoms];isrobot=[n.startswith(('robot0_','mobilebase0_','gripper0_')) for n in names];grasp=any(n.startswith('gripper0_') for n in names) and any(n.startswith('obj_') for n in names);review=any(isrobot) and not all(isrobot) and (not grasp or float(c.dist)<-.01)
                contacts.append(dict(step=len(actions),distance_m=float(c.dist),geoms=geoms,requires_collision_review=bool(review)))
            phase=next_phase
            if errors[-1][0]>.08 or errors[-1][1]>.5:failed='Continuous free-space tracking error too large';break
            if phase>=ref['total']-1e-9:
                # Optional measured correction before entering contact: actual
                # controls only; no injected qpos, no relaxed acceptance.
                if (errors[-1][0]<=.002 and errors[-1][1]<=.02) or catchup_used>=catchup_budget:break
        if phase<ref['total']-1e-9 and failed is None:failed='Continuous free-space clock exceeded execution budget'
    finally:
        for writer in writers:writer.close()
    sim.forward();final_p=sim.data.site_xpos[sid].copy();final_r=sim.data.site_xmat[sid].reshape(3,3).copy();pos_error=float(np.linalg.norm(final_p-ref['positions'][end]));rot_error=float(Rotation.from_matrix(ref['rotations'][end]@final_r.T).magnitude());grip=sim.data.qpos[robot._ref_gripper_joint_pos_indexes['right']].copy();gids=np.asarray(robot._ref_gripper_joint_pos_indexes['right'],int);grip_target=human_states[end,1+gids];grip_error=float(np.max(np.abs(grip-grip_target)))
    if failed is None and (pos_error>float(plan.get('handoff_position_tolerance',.005)) or rot_error>.04):failed='Moving handoff outside operation tracking tolerance'
    if failed is None and grip_error>.003:failed='Moving handoff gripper mismatch'
    np.savez_compressed(output/'transition_actual.npz',actions_env_order=np.asarray(actions),observation_state=np.asarray(obs),sim_state_before_action=np.asarray(states),ee_world_xyz_quat_xyzw=np.asarray(ees),initial_sim_state=initial_sim_state,final_sim_state=sim.get_state().flatten(),reference_ee=np.asarray(targets),reference_time_s=np.asarray(phases),time_scale=np.asarray(rates),source_frame_fraction=np.asarray(source_ids),tracking_error_after_action_m_rad=np.asarray(errors))
    result=dict(status='transition_executed' if failed is None else 'transition_tracking_failed',failure_reason=failed,steps=len(actions),final_position_error_m=pos_error,final_rotation_error_rad=rot_error,human_gripper_max_error_m=grip_error,source_resume_frame=end,virtual_join_frame=k,virtual_join_step=virtual_join_step,deep_contacts=contacts,collision_review_required=any(c['requires_collision_review'] for c in contacts),speed_profile=dict(method='continuous_free_space_clock_with_source_OSC_feedforward',planned_seconds=ref['total'],actual_seconds=len(actions)*dt,minimum_time_scale=float(np.min(rates)) if rates else None,maximum_time_scale=float(np.max(rates)) if rates else None,linear_feedforward_gain=gp,angular_feedforward_gain=gw,fixed_endpoint_settle_steps=catchup_used,reference_phase_aligned=bool(plan.get('reference_phase_aligned',False)),feedback_gain=float(plan.get('continuous_feedback_gain',1.))),handoff_tolerance=dict(position_m=float(plan.get('handoff_position_tolerance',.005)),rotation_rad=.04,scope='moving boundary into retained Human operation; virtual join is not a stop point'),camera_videos={c:str(output/('transition_'+c+'.mp4')) for c in CAMERAS})
    (output/'transition_result.json').write_text(json.dumps(result,indent=2));return result
