"""Track measured Human world-frame EE motion with bounded feedback retiming."""
from pathlib import Path
import json
import numpy as np
import mujoco
import imageio.v2 as imageio
from scipy.spatial.transform import Rotation

CAMERAS = ('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand')
KEYS = ('robot0_base_pos', 'robot0_base_quat', 'robot0_base_to_eef_pos',
        'robot0_base_to_eef_quat', 'robot0_gripper_qpos')


def run_operation(env, human_states, human_actions, output, *, adaptive_interpolation=False,
                  max_interpolation_substeps=4, min_interpolation_substeps=2,
                  interpolate_source_ranges=(), source_range_substeps=2,
                  tracking_max_time_ratio=1.30, tracking_position_tolerance=.005,
                  tracking_rotation_tolerance=.04, source_posture_guidance=False,
                  reference_grasp_contacts=False, tracking_catchup=False,
                  posture_limit_margin_rad=.035, source_actions_only=False, contact_phase_gating=False, object_relative_grasp=False):
    """Preserve source commands once, correcting toward measured next EE poses.

    The original OSC goal supplies feedforward, because an observed next pose
    alone is not the servo goal which produced it. Feedback targets are the
    measured next poses, not an extrapolated command. Added steps hold the
    incremental gripper/torso commands. Every observation is saved BEFORE its
    action. The time budget applies to operation, independently of transition.
    Legacy interpolation options remain accepted for existing launch scripts.
    """
    if interpolate_source_ranges:
        raise ValueError('Manual source-frame interpolation is superseded by bounded measured-trajectory feedback')
    if not 1 <= min_interpolation_substeps <= max_interpolation_substeps:
        raise ValueError('Require 1 <= min_interpolation_substeps <= max_interpolation_substeps')
    if tracking_max_time_ratio < 1 or tracking_position_tolerance <= 0 or tracking_rotation_tolerance <= 0:
        raise ValueError('Invalid tracking budget or tolerances')
    output = Path(output)
    sim, robot = env.sim, env.robots[0]
    cc = robot.composite_controller
    ctrl, split = cc.part_controllers['right'], cc._action_split_indexes
    if ctrl.input_type != 'delta' or ctrl.input_ref_frame != 'base':
        raise RuntimeError('Only PandaOmron base-frame OSC delta control is supported')
    model = sim.model._model
    scratch = mujoco.MjData(model)
    nq, nv = model.nq, model.nv
    if human_states.ndim != 2 or human_states.shape[1] != 1+nq+nv:
        raise ValueError('Incompatible recorded MuJoCo state layout')
    if len(human_states) < len(human_actions) or not len(human_actions):
        raise ValueError('Missing Human states/actions')
    site_id = sim.model.site_name2id(ctrl.ref_name)
    origin_id = sim.model.site_name2id(f'{ctrl.naming_prefix}{ctrl.part_name}_center')
    scale = (np.asarray(ctrl.output_max)-np.asarray(ctrl.output_min))/(np.asarray(ctrl.input_max)-np.asarray(ctrl.input_min))
    midin = (np.asarray(ctrl.input_max)+np.asarray(ctrl.input_min))/2
    midout = (np.asarray(ctrl.output_max)+np.asarray(ctrl.output_min))/2

    measured, source_origins = [], []
    source_contact_depths=[];source_objects=[]
    object_body=env.obj_body_id.get('obj') if object_relative_grasp else None
    if object_relative_grasp and object_body is None:raise ValueError('Missing target object body for grasp correction')
    for state in human_states:
        scratch.time = state[0]
        scratch.qpos[:] = state[1:1+nq]
        scratch.qvel[:] = state[1+nq:]
        mujoco.mj_forward(model, scratch)
        measured.append((scratch.site_xpos[site_id].copy(), scratch.site_xmat[site_id].reshape(3,3).copy()))
        source_origins.append(scratch.site_xmat[origin_id].reshape(3,3).copy())
        if object_relative_grasp:source_objects.append((scratch.xpos[object_body].copy(),scratch.xmat[object_body].reshape(3,3).copy()))
        depths={}
        if reference_grasp_contacts:
            for contact in scratch.contact:
                key=tuple(sorted((int(contact.geom1),int(contact.geom2))))
                depths[key]=min(depths.get(key,0.),float(contact.dist))
        source_contact_depths.append(depths)
    goals = []
    for t, act in enumerate(human_actions):
        p, r = measured[t]
        origin = source_origins[t]
        if act[-1] > 0:
            if t+1 >= len(measured):
                raise ValueError('Base-mode final action requires its recorded next state')
            goal = measured[t+1]
        else:
            delta = (np.clip(act[:6], ctrl.input_min, ctrl.input_max)-midin)*scale+midout
            goal = (p+origin@delta[:3], origin@Rotation.from_rotvec(delta[3:]).as_matrix()@origin.T@r)
        goals.append(goal)

    def pose():
        sim.forward()
        return sim.data.site_xpos[site_id].copy(), sim.data.site_xmat[site_id].reshape(3,3).copy()

    def error(reference):
        p, r = pose()
        return np.array([np.linalg.norm(reference[0]-p),
                         Rotation.from_matrix(reference[1]@r.T).magnitude()])

    def command(target, gripper=0., torso=0.):
        p, r = pose()
        _, origin = cc.get_controller_base_pose('right')
        origin = np.asarray(origin).reshape(3,3)
        delta = np.r_[origin.T@(target[0]-p),
                      Rotation.from_matrix(origin.T@target[1]@r.T@origin).as_rotvec()]
        normalized = (delta-midout)/scale+midin
        ratio = float(np.max(np.abs(normalized)))
        # Uniform scaling preserves translation/rotation command direction.
        action = np.zeros(env.action_dim, dtype=float)
        a,b = split['right']; action[a:b] = normalized/max(1.,ratio)
        a,b = split['right_gripper']; action[a:b] = gripper
        a,b = split['torso']; action[a:b] = torso
        action[-1] = -1.
        return action, ratio

    writers = [imageio.get_writer(str(output/(c+'.mp4')), fps=float(env.control_freq),
               codec='libx264', pixelformat='yuv420p', macro_block_size=2,
               ffmpeg_params=['-threads','1','-preset','veryfast']) for c in CAMERAS]
    actions=[]; obs=[]; simstates=[]; ee=[]; successes=[]; contacts=[]
    source_ids=[]; substeps=[]; ratios=[]; source_ratios=[]; refs=[]; errors=[]; kinds=[]
    source_errors=[]; corrected_sources=set(); saturated_sources=set(); added=0
    n = len(human_actions)
    extra_budget = int(np.floor(n*(tracking_max_time_ratio-1)+1e-8))
    max_corrections = max_interpolation_substeps-1
    success = False

    posture=None
    if source_posture_guidance:
        from source_posture import SourcePosture
        posture=SourcePosture(ctrl,model,robot.robot_model.joints,limit_margin_rad=posture_limit_margin_rad)
    def step(action, t, substep, ratio, reference, initial_ratio):
        p,r = pose()
        ob = env._get_observations(force_update=True)
        obs.append(np.concatenate([np.asarray(ob[k]) for k in KEYS]))
        simstates.append(sim.get_state().flatten().copy())
        ee.append(np.r_[p,Rotation.from_matrix(r).as_quat()])
        for writer, cam in zip(writers,CAMERAS):
            writer.append_data(np.asarray(sim.render(width=256,height=256,camera_name=cam))[::-1])
        if posture is not None:posture.update(human_states[min(t+1,len(human_states)-1)])
        env.step(action)
        actions.append(action); source_ids.append(t); substeps.append(substep)
        ratios.append(ratio); source_ratios.append(initial_ratio); kinds.append('source' if substep==0 else 'feedback')
        refs.append(np.r_[reference[0],Rotation.from_matrix(reference[1]).as_quat()] if reference is not None else np.full(7,np.nan))
        errors.append(error(reference) if reference is not None else np.full(2,np.nan))
        success = bool(env._check_success()); successes.append(success)
        for i in range(sim.data.ncon):
            c = sim.data.contact[i]
            if float(c.dist) >= -.005: continue
            parts=[]
            for gid in (int(c.geom1),int(c.geom2)):
                body=int(sim.model.geom_bodyid[gid])
                parts.append({'geom_id':gid,'geom':sim.model.geom_id2name(gid),'body':sim.model.body_id2name(body)})
            names=[x['body'] or '' for x in parts]
            robots=[s.startswith(('robot0_','mobilebase0_','gripper0_')) for s in names]
            grasp=any(s.startswith('gripper0_') for s in names) and any(s.startswith('obj_') for s in names)
            pair=tuple(sorted((int(c.geom1),int(c.geom2))))
            source_depth=source_contact_depths[min(t+1,len(source_contact_depths)-1)].get(pair,0.)
            allowed_depth=min(-.010,source_depth-.001) if reference_grasp_contacts and grasp else -.010
            contacts.append({'step':len(actions)-1,'source_frame_index':t,'interpolation_step_index':substep,
                             'geoms':parts,'distance_m':float(c.dist),'expected_gripper_target':grasp,
                             'source_same_pair_distance_m':source_depth,'allowed_grasp_distance_m':allowed_depth,
                             'requires_collision_review':bool(any(robots) and not all(robots) and (not grasp or float(c.dist)<allowed_depth))})
        return success

    closing=np.flatnonzero(human_actions[:,6]>.1)
    close_frame=int(closing[0]) if len(closing) else None
    frozen_correction=None;object_corrections=[]
    try:
        for t, goal in enumerate(goals):
            current_reference=measured[t];next_reference=measured[t+1] if t+1<len(measured) else None
            if object_relative_grasp and close_frame is not None and close_frame-12<=t<close_frame+20:
                if t<=close_frame or frozen_correction is None:
                    sim.forward();center,ref_rotation=source_objects[t]
                    shift=sim.data.body_xpos[object_body].copy()-center
                    delta_r=sim.data.body_xmat[object_body].reshape(3,3).copy()@ref_rotation.T
                    rv=Rotation.from_matrix(delta_r).as_rotvec()
                    if np.linalg.norm(shift)>.025 or np.linalg.norm(rv)>.7:raise RuntimeError('Object grasp correction exceeds bounded 2.5cm/0.7rad envelope')
                    frozen_correction=(center.copy(),shift.copy(),rv.copy())
                center,shift,rv=frozen_correction
                w=1. if t<=close_frame else max(0.,1.-(t-close_frame)/20.)
                w=w*w*w*(10-15*w+6*w*w)
                rot=Rotation.from_rotvec(w*rv).as_matrix()
                def corrected(pose):return (center+rot@(pose[0]-center)+w*shift,rot@pose[1])
                goal=corrected(goal);current_reference=corrected(current_reference)
                if next_reference is not None:next_reference=corrected(next_reference)
                object_corrections.append(dict(source_frame=t,weight=w,translation_m=shift.tolist(),rotation_vector_rad=rv.tolist()))
            if contact_phase_gating and human_actions[t,6]>.1 and (t==0 or human_actions[t-1,6]<=.1):
                # Arrive at the recorded pre-close pose before applying the
                # closing command; charge every real step to the same budget.
                for extra in range(3):
                    e=error(current_reference)
                    if (e[0]<=.002 and e[1]<=.02) or added>=extra_budget:break
                    corrective,rr=command(current_reference)
                    step(corrective,t,extra+1,rr,current_reference,rr);added+=1
            reference = next_reference
            action, initial_ratio = command(goal, human_actions[t,6], human_actions[t,10])
            if source_actions_only:
                action=np.asarray(human_actions[t],dtype=float).copy();initial_ratio=1.
            if initial_ratio > 1+1e-8: saturated_sources.add(t)
            success = step(action,t,0,initial_ratio,reference,initial_ratio)
            if reference is not None and not source_actions_only:
                # Distribute available correction time through the episode,
                # reserving future budget instead of consuming it at the start.
                available = extra_budget if tracking_catchup else min(extra_budget, int(np.ceil((t+1)*(tracking_max_time_ratio-1))))
                for substep in range(1,max_corrections+1):
                    e = errors[-1]
                    if success or (e[0] <= tracking_position_tolerance and e[1] <= tracking_rotation_tolerance) or added >= available:
                        break
                    action, ratio = command(reference)
                    success = step(action,t,substep,ratio,reference,initial_ratio)
                    corrected_sources.add(t); added += 1
                source_errors.append(errors[-1])
            if success: break
    finally:
        for writer in writers: writer.close()
    source_errors=np.asarray(source_errors)
    def quantiles(column):
        return {'median':float(np.median(column)), 'p95':float(np.quantile(column,.95)), 'max':float(np.max(column))} if len(column) else None
    np.savez_compressed(output/'operation_actual.npz', actions_env_order=np.asarray(actions),
        observation_state=np.asarray(obs), sim_state_before_action=np.asarray(simstates),
        ee_world_xyz_quat_xyzw=np.asarray(ee), success_after_action=np.asarray(successes),
        source_frame_index=np.asarray(source_ids), interpolation_step_index=np.asarray(substeps),
        requested_action_ratio=np.asarray(ratios), source_initial_request_ratio=np.asarray(source_ratios),
        reference_next_ee_world_xyz_quat_xyzw=np.asarray(refs), tracking_error_after_action_m_rad=np.asarray(errors),
        action_kind=np.asarray(kinds), final_sim_state=sim.get_state().flatten())
    consumed = source_ids[-1]+1 if source_ids else 0
    result = {'success_ever':bool(any(successes)), 'steps':len(actions), 'source_steps':n,
        'saturated_action_steps':len(saturated_sources),
        'interpolation':{'enabled':True,'method':'measured_ee_feedback_with_original_osc_feedforward',
            'inserted_control_steps':added,'corrected_source_frames':len(corrected_sources),
            'max_extra_steps':extra_budget,'max_operation_time_ratio':tracking_max_time_ratio},
        'tracking':{'position_tolerance_m':tracking_position_tolerance,'rotation_tolerance_rad':tracking_rotation_tolerance,
            'position_error_after_source_m':quantiles(source_errors[:,0]) if len(source_errors) else None,
            'rotation_error_after_source_rad':quantiles(source_errors[:,1]) if len(source_errors) else None,
            'source_frames_consumed':consumed,'time_ratio_to_consumed_source':len(actions)/consumed if consumed else None,
            'source_reference':'FK of recorded state[t+1]; last unavailable next state is explicitly masked',
            'gripper_torso':'Source incremental commands applied once on the normal source step; zero on feedback steps'},
        'object_relative_grasp':object_relative_grasp,'object_corrections':object_corrections,'contact_phase_gating':contact_phase_gating,'source_actions_only':source_actions_only,'posture_limit_margin_rad':posture_limit_margin_rad,'source_posture_guidance':source_posture_guidance,'reference_grasp_contacts':reference_grasp_contacts,'tracking_catchup':tracking_catchup,
        'target_methods':{'recorded_osc_goal_with_measured_trajectory_feedback':n},
        'deep_contacts':contacts,'collision_review_required':any(c['requires_collision_review'] for c in contacts),
        'operation_data':str(output/'operation_actual.npz'),
        'videos':{c:str(output/(c+'.mp4')) for c in CAMERAS},
        'note':'Only actually executed actions/states/images are saved. Task success is evaluated in the simulator; contacts remain diagnostic.'}
    (output/'operation_result.json').write_text(json.dumps(result,indent=2))
    return result
