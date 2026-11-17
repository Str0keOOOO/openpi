"""Execute and record a real OSC transition from Anchor to Human initial EE pose."""
from pathlib import Path
import json
import numpy as np
import imageio.v2 as imageio
import mujoco
from scipy.spatial.transform import Rotation, Slerp

CAMERAS=('robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand')


def human_speed_profile(sim, site_id, states, frequency):
    """Measure moving-frame source EE speeds using this episode's actual FK."""
    if states is None or len(states)<3:
        raise ValueError('Real Human states are required for transition speed matching')
    model=sim.model._model; scratch=mujoco.MjData(model);positions=[];rotations=[]
    for state in states:
        scratch.qpos[:]=state[1:1+model.nq]
        scratch.qvel[:]=state[1+model.nq:1+model.nq+model.nv]
        mujoco.mj_forward(model,scratch)
        positions.append(scratch.site_xpos[site_id].copy())
        rotations.append(scratch.site_xmat[site_id].reshape(3,3).copy())
    linear=np.linalg.norm(np.diff(positions,axis=0),axis=1)*frequency
    rotations=np.asarray(rotations)
    angular=Rotation.from_matrix(rotations[1:]@rotations[:-1].transpose(0,2,1)).magnitude()*frequency
    moving=linear[linear>.005];turning=angular[angular>.03]
    # Robust active-motion speed, bounded for controller tracking rather than
    # matching noisy peaks or pauses in the demonstration.
    v=float(np.quantile(moving,.75)) if len(moving) else .04
    w=float(np.quantile(turning,.75)) if len(turning) else .2
    return dict(source_linear_p75_m_s=v,source_angular_p75_rad_s=w,
                linear_m_s=float(np.clip(v,.04,.25)),angular_rad_s=float(np.clip(w,.2,1.5)),
                source_frames=len(states),frequency_hz=frequency)


def run_transition(env, initial_sim_state, start_pos, start_rot, human_pos, human_rot, output, source_gripper_qpos, human_states=None, plan=None):
    if plan is None:raise ValueError("Smooth join requires a selected pre-contact plan")
    from smooth_join_plan import curve
    output=Path(output)
    sim=env.sim
    sim.set_state_from_flattened(initial_sim_state)
    sim.forward()
    robot=env.robots[0]
    scene_initial_qpos=np.asarray(sim.data.qpos).copy()
    cc=robot.composite_controller
    # This initializes a new controller for a new Anchor start. Human action
    # replay itself remains on the separate canonical reset_to-only path.
    cc.update_state()
    for controller in cc.part_controllers.values():
        controller.update(force=True)
    cc.reset()
    ctrl=cc.part_controllers['right']
    if ctrl.input_type != 'delta' or ctrl.input_ref_frame != 'base':
        raise RuntimeError(f'Unsupported OSC mode {ctrl.input_type}, {ctrl.input_ref_frame}')
    split=cc._action_split_indexes
    arm_start,arm_end=split['right'];grip_start,grip_end=split['right_gripper']
    if arm_end-arm_start != 6 or grip_end-grip_start != 1:
        raise RuntimeError(f'Unexpected PandaOmron action mapping: {split}')
    sid=sim.model.site_name2id(ctrl.ref_name)
    rots=Slerp([0.,1.],Rotation.from_matrix(np.stack([start_rot,human_rot])))
    writer=imageio.get_writer(str(output/'transition_preview.mp4'),fps=float(env.control_freq),codec='libx264',pixelformat='yuv420p',macro_block_size=2,ffmpeg_params=['-threads','1','-preset','veryfast'])
    camera_writers=[imageio.get_writer(str(output/('transition_'+c+'.mp4')),fps=float(env.control_freq),codec='libx264',pixelformat='yuv420p',macro_block_size=2,ffmpeg_params=['-threads','1','-preset','veryfast']) for c in CAMERAS]
    actions=[];states=[];simstates=[];ee=[];contact_records=[];failed=None
    keys=('robot0_base_pos','robot0_base_quat','robot0_base_to_eef_pos','robot0_base_to_eef_quat','robot0_gripper_qpos')
    try:
        speed = human_speed_profile(sim, sid, human_states, float(env.control_freq))
        distance=float(np.linalg.norm(human_pos-start_pos))
        angle=float(Rotation.from_matrix(human_rot@start_rot.T).magnitude())
        dt=1./float(env.control_freq)
        duration=float(plan['duration_s'])
        speed.update(linear_m_s=plan['source_linear_speed_m_s'],angular_rad_s=plan['source_angular_speed_rad_s'])
        scheduled=max(2,int(np.ceil(duration/dt)))
        max_steps=scheduled+max(40,scheduled)
        progress=0.;paused=0;max_pos_error=0.;max_rot_error=0.;reached=False
        for step in range(max_steps):
            sim.forward()
            current_pos=np.asarray(sim.data.site_xpos[sid]).copy()
            current_rot=np.asarray(sim.data.site_xmat[sid]).reshape(3,3).copy()
            final_p=float(np.linalg.norm(human_pos-current_pos))
            final_r=float(Rotation.from_matrix(human_rot@current_rot.T).magnitude())
            if progress>=1. and final_p<.003 and final_r<.04:
                reached=True
                break
            # Advance a continuous smooth target; pause only if measured tracking
            # lags materially. There is no wait at intermediate waypoints.
            previous_positions,previous_rotations=curve(start_pos,start_rot,human_pos,human_rot,np.asarray(plan['terminal_velocity_world_m_s']),np.asarray(plan['terminal_angular_velocity_world_rad_s']),duration,[progress])
            previous_pos,previous_rot=previous_positions[0],previous_rotations[0]
            lag_p=float(np.linalg.norm(previous_pos-current_pos))
            lag_r=float(Rotation.from_matrix(previous_rot@current_rot.T).magnitude())
            if lag_p<=.015 and lag_r<=.12:
                progress=min(1.,progress+1./scheduled)
            else:
                paused+=1
            targets,target_rotations=curve(start_pos,start_rot,human_pos,human_rot,np.asarray(plan['terminal_velocity_world_m_s']),np.asarray(plan['terminal_angular_velocity_world_rad_s']),duration,[progress])
            target_pos,target_rot=targets[0],target_rotations[0]
            _,origin_rot=cc.get_controller_base_pose('right')
            origin_rot=np.asarray(origin_rot).reshape(3,3)
            pos_delta=origin_rot.T@(target_pos-current_pos)
            rot_delta=Rotation.from_matrix(origin_rot.T@target_rot@current_rot.T@origin_rot).as_rotvec()
            max_pos_error=max(max_pos_error,float(np.linalg.norm(pos_delta)))
            max_rot_error=max(max_rot_error,float(np.linalg.norm(rot_delta)))
            feed_pos=origin_rot.T@(target_pos-previous_pos)
            feed_rot=origin_rot.T@Rotation.from_matrix(target_rot@previous_rot.T).as_rotvec()
            delta=np.r_[1.5*pos_delta+.75*feed_pos,1.5*rot_delta+.75*feed_rot]
            scale=(np.asarray(ctrl.output_max)-np.asarray(ctrl.output_min))/(np.asarray(ctrl.input_max)-np.asarray(ctrl.input_min))
            midin=(np.asarray(ctrl.input_max)+np.asarray(ctrl.input_min))/2
            midout=(np.asarray(ctrl.output_max)+np.asarray(ctrl.output_min))/2
            action=np.zeros(env.action_dim,dtype=np.float64)
            action[arm_start:arm_end]=np.clip((delta-midout)/scale+midin,-1,1)
            # PandaGripper -1 keeps opening; zero preserves its current
            # velocity target from the reset. Initial width is checked
            # against the recorded Human width at the transition end.
            action[grip_start:grip_end]=float(plan.get('approach_gripper_action',0.))
            action[-1]=-1.0
            ob=env._get_observations(force_update=True)
            states.append(np.concatenate([np.asarray(ob[k]) for k in keys]))
            simstates.append(sim.get_state().flatten().copy())
            ee.append(np.r_[current_pos,Rotation.from_matrix(current_rot).as_quat()])
            images=[np.asarray(sim.render(width=256,height=256,camera_name=c))[::-1] for c in CAMERAS]
            writer.append_data(np.concatenate(images,axis=1))
            for camera_writer,image in zip(camera_writers,images): camera_writer.append_data(image)
            env.step(action);actions.append(action)
            for i in range(sim.data.ncon):
                contact=sim.data.contact[i]
                if float(contact.dist)<-.005:
                    parts=[]
                    for gid in (int(contact.geom1),int(contact.geom2)):
                        body=int(sim.model.geom_bodyid[gid])
                        parts.append({'geom_id':gid,'geom':sim.model.geom_id2name(gid),
                                      'body':sim.model.body_id2name(body)})
                    names=[(x['body'] or '') for x in parts]
                    is_robot=[name.startswith(('robot0_','mobilebase0_','gripper0_')) for name in names]
                    expected_grasp=any(name.startswith('gripper0_') for name in names) and any(name.startswith('obj_') for name in names)
                    review=(any(is_robot) and not all(is_robot) and
                            (not expected_grasp or float(contact.dist)<-.010))
                    contact_records.append({'step':len(actions),'geoms':parts,
                                            'distance_m':float(contact.dist),
                                            'requires_collision_review':review})
        sim.forward()
        reached=(float(np.linalg.norm(np.asarray(sim.data.site_xpos[sid])-human_pos))<.003 and
                 float(Rotation.from_matrix(human_rot@np.asarray(sim.data.site_xmat[sid]).reshape(3,3).T).magnitude())<.04)
        if not reached:
            failed='Continuous transition exhausted bounded tracking budget'
        speed.update(method='precontact_quintic_tangent_join',scheduled_steps=scheduled,
                     executed_steps=len(actions),lag_pause_steps=paused,
                     planned_seconds=scheduled*dt,actual_seconds=len(actions)*dt,
                     max_target_position_error_m=max_pos_error,max_target_rotation_error_rad=max_rot_error)
    finally:
        writer.close()
        for camera_writer in camera_writers: camera_writer.close()
    sim.forward()
    final_pos=np.asarray(sim.data.site_xpos[sid]).copy()
    final_rot=np.asarray(sim.data.site_xmat[sid]).reshape(3,3).copy()
    pos_err=float(np.linalg.norm(final_pos-human_pos))
    rot_err=float(Rotation.from_matrix(human_rot@final_rot.T).magnitude())
    actual_grip=np.asarray(sim.data.qpos[robot._ref_gripper_joint_pos_indexes['right']]).copy()
    grip_err=float(np.max(np.abs(actual_grip-source_gripper_qpos)))
    if failed is None and grip_err > 0.003:
        failed=f'Gripper width differs from Human initial state by {grip_err:.6f} m'
    # Human frame-zero environment is the operation start target.  Check each
    # non-robot joint in its own units; a quaternion must not be compared as a
    # scalar joint position.  Object motion invalidates the intended handoff.
    model=sim.model._model
    robot_qpos_ids=set(int(x) for x in robot._ref_all_joint_pos_indexes)
    for arm in robot.arms:
        if robot.has_gripper[arm]:
            robot_qpos_ids.update(int(x) for x in robot._ref_gripper_joint_pos_indexes[arm])
    qfinal=np.asarray(sim.data.qpos)
    scene_drift=[]
    for jid in range(model.njnt):
        start=int(model.jnt_qposadr[jid]);stop=int(model.jnt_qposadr[jid+1]) if jid+1<model.njnt else model.nq
        if any(k in robot_qpos_ids for k in range(start,stop)): continue
        before=scene_initial_qpos[start:stop];after=qfinal[start:stop]
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,jid) or str(jid)
        typ=int(model.jnt_type[jid]); item={'joint':name,'type':typ}
        if typ==int(mujoco.mjtJoint.mjJNT_FREE):
            item['translation_m']=float(np.linalg.norm(after[:3]-before[:3]))
            q0=before[3:7][[1,2,3,0]];q1=after[3:7][[1,2,3,0]]
            item['rotation_rad']=float((Rotation.from_quat(q1)*Rotation.from_quat(q0).inv()).magnitude())
            significant=item['translation_m']>.003 or item['rotation_rad']>np.deg2rad(3)
        elif typ==int(mujoco.mjtJoint.mjJNT_BALL):
            q0=before[[1,2,3,0]];q1=after[[1,2,3,0]]
            item['rotation_rad']=float((Rotation.from_quat(q1)*Rotation.from_quat(q0).inv()).magnitude())
            significant=item['rotation_rad']>np.deg2rad(3)
        else:
            item['delta']=float(after[0]-before[0])
            threshold=.003 if typ==int(mujoco.mjtJoint.mjJNT_SLIDE) else np.deg2rad(3)
            significant=abs(item['delta'])>threshold
        if significant: scene_drift.append(item)
    # Deep contacts are diagnostics during the trajectory. The official
    # RoboCasa collision predicate is applied to the exact Anchor start pose
    # in preflight.py; applying it to every operation frame would also reject
    # intentional gripper/object contacts.
    np.savez_compressed(output/'transition_actual.npz',actions_env_order=np.asarray(actions),observation_state=np.asarray(states),sim_state_before_action=np.asarray(simstates),ee_world_xyz_quat_xyzw=np.asarray(ee),initial_sim_state=initial_sim_state,final_sim_state=sim.get_state().flatten(),source_human_gripper_qpos=source_gripper_qpos,final_gripper_qpos=actual_grip)
    result={'handoff_plan':plan,'speed_profile':speed,'status':'transition_executed' if failed is None else 'transition_tracking_failed','failure_reason':failed,'steps':len(actions),'final_position_error_m':pos_err,'final_rotation_error_rad':rot_err,'human_gripper_max_error_m':grip_err,'scene_drift':scene_drift,'scene_drift_thresholds_diagnostic_only':{'free_translation_m':.003,'rotation_deg':3,'slide_m':.003},'deep_contacts':contact_records,'collision_review_required':any(c['requires_collision_review'] for c in contact_records),'contact_thresholds_diagnostic_only':{'record_if_penetration_exceeds_m':.005,'gripper_target_review_if_exceeds_m':.010},'video':str(output/'transition_preview.mp4'),'camera_videos':{c:str(output/('transition_'+c+'.mp4')) for c in CAMERAS},'data':str(output/'transition_actual.npz'),'note':'Natural scene-object motion during the handoff is measured and retained for diagnosis, but it does not abort the rollout. Official RoboCasa collision rejection is applied at the exact Anchor start pose; EE tracking and task success remain required.'}
    (output/'transition_result.json').write_text(json.dumps(result,indent=2))
    return result
