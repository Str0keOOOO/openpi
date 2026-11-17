"""Generate episode-initial, two-agentview SHARP parallax teacher caches.

The script intentionally refuses unreviewed anchors unless --provisional is
given. Provisional results are written to a separate cache and are never
selected by the formal training config.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from examples.robocasa_spatial.geometry import masked_average_pool, parallax
from examples.robocasa_spatial.teacher_cache import CAMERAS, save

DATA_ROOT = Path('/data4/xuchenfei/robocasa365/third_party/robocasa/datasets/v1.0/pretrain/atomic')
AUDIT_ROOT = Path('/data4/xuchenfei/robocasa365/output/anchor_visual_audit')
TASKS = ('PickPlaceSinkToCounter', 'OpenDrawer', 'OpenStandMixerHead', 'PickPlaceCounterToCabinet')
CAMERA_KEYS = {
    'robot0_agentview_left': 'observation.images.robot0_agentview_left',
    'robot0_agentview_right': 'observation.images.robot0_agentview_right',
}
MUJOCO_TO_OPENCV = np.diag([1., -1., -1., 1.])


def read_frame(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'cannot open RGB video: {path}')
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f'cannot decode frame 0: {path}')
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def camera_calibration(sim, name: str, height: int, width: int):
    """Return K and T_world_opencv_camera from live MuJoCo camera state."""
    cid = sim.model.camera_name2id(name)
    fovy = float(sim.model.cam_fovy[cid])
    focal = height / (2.0 * math.tan(math.radians(fovy) / 2.0))
    k = np.array([[focal, 0., width / 2.], [0., focal, height / 2.], [0., 0., 1.]], dtype=np.float64)
    t_world_mj = np.eye(4)
    t_world_mj[:3, :3] = np.asarray(sim.data.cam_xmat[cid]).reshape(3, 3)
    t_world_mj[:3, 3] = np.asarray(sim.data.cam_xpos[cid])
    return k, t_world_mj @ MUJOCO_TO_OPENCV, fovy


def robot_geom_ids(sim) -> np.ndarray:
    """Resolve robot visual/collision geoms from compiled model names."""
    prefixes = ('robot0_', 'mobilebase0_', 'gripper0_', 'panda0_', 'panda_')
    body_ids = set()
    for bid in range(sim.model.nbody):
        name = sim.model.body_id2name(bid) or ''
        if name.startswith(prefixes):
            body_ids.add(bid)
    # Include descendants of robot roots, even if a model gives a child a neutral name.
    changed = True
    while changed:
        changed = False
        for bid in range(1, sim.model.nbody):
            if int(sim.model.body_parentid[bid]) in body_ids and bid not in body_ids:
                body_ids.add(bid); changed = True
    ids = []
    for gid in range(sim.model.ngeom):
        name = sim.model.geom_id2name(gid) or ''
        if int(sim.model.geom_bodyid[gid]) in body_ids or name.startswith(prefixes):
            ids.append(gid)
    if not ids:
        raise RuntimeError('compiled MuJoCo model contains no identifiable robot geoms')
    return np.asarray(ids, dtype=np.int32)


def robot_mask(sim, camera: str, width: int, height: int, geom_ids: np.ndarray) -> np.ndarray:
    seg = np.asarray(sim.render(camera_name=camera, width=width, height=height, segmentation=True))
    if seg.shape != (height, width, 2):
        raise RuntimeError(f'unexpected MuJoCo segmentation shape for {camera}: {seg.shape}')
    # The dataset RGB is OpenCV/top-origin, while raw MuJoCo rendering is bottom-origin.
    return np.isin(seg[::-1, :, 1], geom_ids)


def _env_and_pose(task, episode, meta, xml_path, state0, env_kwargs, anchor_pose):
    import robocasa  # noqa: F401 - register RoboCasa tasks
    import robosuite as suite
    from robosuite.utils import transform_utils as T

    kw = dict(env_kwargs)
    kw.update(has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False,
              camera_names=list(CAMERA_KEYS), camera_widths=256, camera_heights=256,
              camera_depths=False, render_gpu_device_id=-1, seed=0)
    env = suite.make(**kw)
    try:
        env.set_ep_meta(meta)
        env.reset()
        xml = gzip.open(xml_path, 'rt').read()
        env.reset_from_xml_string(env.edit_model_xml(xml))
        env.sim.reset()
        env.sim.set_state_from_flattened(state0)
        env.sim.forward()

        source_pose = {}; geom_ids=robot_geom_ids(env.sim)
        h, w = 256, 256
        for cam in CAMERA_KEYS:
            k, t, fovy = camera_calibration(env.sim, cam, h, w)
            source_pose[cam] = (k, t, fovy, robot_mask(env.sim, cam, w, h, geom_ids))

        # Change only the mobile-base planar pose; retain all saved arm/gripper joints.
        anchor_xyz, anchor_rpy = anchor_pose
        anchor_yaw = float(anchor_rpy[2])
        anchor_ori = np.asarray(anchor_rpy, dtype=float)
        inv = T.matrix_inverse(T.euler2mat(anchor_ori))
        local = np.matmul(inv, np.r_[np.asarray(anchor_xyz[:2], dtype=float), 0.0])
        undo = np.matmul(inv, [-10., -10., 0.])
        xname, yname = 'mobilebase0_joint_mobile_forward', 'mobilebase0_joint_mobile_side'
        if (env.sim.model.jnt_axis[env.sim.model.joint_name2id(xname)] == np.array([0, 1, 0])).all():
            xname, yname = yname, xname
        env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(xname)] = undo[0] + local[0]
        env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(yname)] = undo[1] + local[1]
        env.sim.data.qpos[env.sim.model.get_joint_qpos_addr('mobilebase0_joint_mobile_yaw')] = anchor_yaw - anchor_rpy[2]
        env.sim.forward()

        reference_pose = {}
        for cam in CAMERA_KEYS:
            k, t, fovy = camera_calibration(env.sim, cam, h, w)
            reference_pose[cam] = (k, t, fovy, robot_mask(env.sim, cam, w, h, geom_ids))
        return env, source_pose, reference_pose
    except Exception:
        env.close()
        raise


def load_selected_anchors(anchor_csv: Path, review_csv: Path | None, provisional: bool):
    with anchor_csv.open(newline='') as f:
        candidates = {(r['task'], int(r['episode_index'])): r for r in csv.DictReader(f)}
    reviewed = {}
    if review_csv and review_csv.is_file():
        with review_csv.open(newline='') as f:
            for r in csv.DictReader(f):
                reviewed[(r['task'], int(r['episode_index']))] = str(r.get('review_status', '')).strip().lower()
    chosen = {}
    for key, row in candidates.items():
        if row.get('status') != 'rendered_candidate':
            continue
        if provisional:
            chosen[key] = row
        elif reviewed.get(key) == 'accepted':
            chosen[key] = row
    return chosen, reviewed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--anchor-csv', type=Path, default=AUDIT_ROOT/'anchor_candidates.csv')
    p.add_argument('--review-csv', type=Path, default=AUDIT_ROOT/'review_template.csv')
    p.add_argument('--sharp-checkpoint', type=Path, default=Path('/data4/xuchenfei/robocasa365/model/sharp/sharp_2572gikvuh.pt'))
    p.add_argument('--output', type=Path)
    p.add_argument('--provisional', action='store_true', help='allow unreviewed poses; output is explicitly non-training')
    p.add_argument('--limit', type=int, default=0, help='debug cap; 0 means all selected episodes')
    a = p.parse_args()
    out = a.output or AUDIT_ROOT/'teacher_cache'/('provisional' if a.provisional else 'formal')
    if a.provisional and 'provisional' not in str(out):
        p.error('--provisional requires an output path containing "provisional"')
    if not a.provisional and 'formal' not in str(out):
        p.error('formal runs require an output path containing "formal"')
    if not a.sharp_checkpoint.is_file():
        p.error(f'SHARP checkpoint missing: {a.sharp_checkpoint}')
    chosen, reviewed = load_selected_anchors(a.anchor_csv, a.review_csv, a.provisional)
    if a.limit:
        chosen = dict(list(sorted(chosen.items()))[:a.limit])
    if not chosen:
        raise RuntimeError('no eligible anchors; formal cache requires review_status=accepted, provisional mode requires rendered candidates')

    import torch
    from sharp.cli.predict import predict_image
    from sharp.models import PredictorParams, create_predictor
    from sharp.utils.gsplat import GSplatRenderer
    if not torch.cuda.is_available():
        raise RuntimeError('SHARP teacher generation requires CUDA + gsplat; this runtime exposes CPU only')
    device = torch.device('cuda')
    state_dict = torch.load(a.sharp_checkpoint, map_location='cpu', weights_only=True)
    predictor = create_predictor(PredictorParams()); predictor.load_state_dict(state_dict); predictor.eval().to(device)
    checkpoint_sha = _sha256(a.sharp_checkpoint)
    renderer = GSplatRenderer()
    out.mkdir(parents=True, exist_ok=True)
    log_path = out/'failures.jsonl'
    summary = {'mode':'provisional' if a.provisional else 'formal','selected':len(chosen),'completed':0,'skipped_cache':0,'failed':0}

    for (task, eid), row in sorted(chosen.items()):
        path = out/task/f'episode_{eid:06d}.npz'
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as f:
                    m=json.loads(str(f['metadata']))
                if m.get('cache_version') == 'episode-agentview-parallax-v1' and m.get('mode') == summary['mode']:
                    summary['skipped_cache'] += 1; continue
            except Exception:
                pass
        env = None
        try:
            root = next((DATA_ROOT/task).glob('*/lerobot'))
            ep = root/'extras'/f'episode_{eid:06d}'
            meta = json.loads((ep/'ep_meta.json').read_text())
            state0 = np.load(ep/'states.npz')['states'][0]
            human_rpy = np.asarray(meta['init_robot_base_ori'], dtype=float)
            anchor_xyz = np.array([float(row['anchor_x']), float(row['anchor_y']), float(meta['init_robot_base_pos'][2])])
            anchor_rpy = human_rpy.copy(); anchor_rpy[2] = math.radians(float(row['anchor_yaw_deg']))
            env_args=json.loads((root/'extras/dataset_meta.json').read_text())['env_args']
            env, source_pose, reference_pose = _env_and_pose(task,eid,meta,ep/'model.xml.gz',state0,env_args['env_kwargs'],(anchor_xyz,anchor_rpy))

            latents={}; masks={}; camera_meta={}; pixel_robot_masks_current={};pixel_robot_masks_reference={};pixel_flow_masks={}
            for cam, video_key in CAMERA_KEYS.items():
                video=root/'videos'/f'chunk-{eid//1000:03d}'/video_key/f'episode_{eid:06d}.mp4'
                rgb=read_frame(video); h,w=rgb.shape[:2]
                k,t_world_src,fovy,robot_src=source_pose[cam]
                _,t_world_ref,_,robot_ref=reference_pose[cam]
                f=float(k[0,0])
                gauss=predict_image(predictor,rgb,f,device)
                k_t=torch.as_tensor(k,dtype=torch.float32,device=device)[None]
                identity=torch.eye(4,dtype=torch.float32,device=device)[None]
                # SHARP Gaussians live in current OpenCV camera coordinates.
                ref_from_src=np.linalg.inv(t_world_ref)@t_world_src
                view_ref=torch.as_tensor(ref_from_src,dtype=torch.float32,device=device)[None]
                with torch.no_grad():
                    cur=renderer(gauss,identity,k_t,w,h)
                    ref=renderer(gauss,view_ref,k_t,w,h)
                dcur=cur.depth[0,0].detach().cpu().numpy(); acur=cur.alpha[0,0].detach().cpu().numpy()
                dref=ref.depth[0,0].detach().cpu().numpy(); aref=ref.alpha[0,0].detach().cpu().numpy()
                valid_cur=np.isfinite(dcur)&(dcur>0)&(acur>0.5)&(~robot_src)
                valid_ref=np.isfinite(dref)&(dref>0)&(aref>0.5)&(~robot_ref)
                # Pass camera poses as source-frame transforms to the tested geometry routine.
                source_from_ref=np.linalg.inv(ref_from_src)
                field,valid=parallax(dcur,k,np.eye(4),source_from_ref,valid_cur,dref,reference_valid=valid_ref)
                z,m=masked_average_pool(field,valid,size=4)
                latents[cam]=z;masks[cam]=m
                pixel_robot_masks_current[cam]=robot_src
                pixel_robot_masks_reference[cam]=robot_ref
                pixel_flow_masks[cam]=valid
                camera_meta[cam]={'fovy_deg':float(fovy),'intrinsics':k.tolist(),'world_from_current_opencv':t_world_src.tolist(),
                    'world_from_reference_opencv':t_world_ref.tolist(),'robot_mask_fraction_current':float(robot_src.mean()),
                    'robot_mask_fraction_reference':float(robot_ref.mean()),'valid_flow_fraction':float(valid.mean()),
                    'sharp_depth_median':float(np.median(dcur[valid_cur])) if valid_cur.any() else None,
                    'reference_depth_median':float(np.median(dref[valid_ref])) if valid_ref.any() else None}
                del gauss,cur,ref
                torch.cuda.empty_cache()
            metadata={'task':task,'episode':eid,'frame':0,'sharp_version':'apple-ml-sharp-0.1-local',
                'sharp_checkpoint_sha256':checkpoint_sha,'anchor_source':str(row.get('anchor_source','RoboCasa candidate')),
                'anchor_review_status':'accepted' if not a.provisional else 'unreviewed_provisional',
                'calibration_id':'robocasa_saved_xml_camera_fovy_v1','cache_version':'episode-agentview-parallax-v1',
                'teacher_scope':'episode_initial_static_scene','mode':summary['mode'],'anchor_xyz_world':anchor_xyz.tolist(),
                'anchor_rpy_world_rad':anchor_rpy.tolist(),'parallax_direction':'current_pixel_to_anchor_projection_minus_current_uv',
                'flow_normalization':'du/image_width,dv/image_height','pooling':'masked adaptive average 4x4 per agentview',
                'cameras':camera_meta}
            save(path,latents,masks,metadata,pixel_robot_masks_current=pixel_robot_masks_current,
                 pixel_robot_masks_reference=pixel_robot_masks_reference,pixel_flow_masks=pixel_flow_masks)
            summary['completed']+=1
        except Exception as exc:
            summary['failed']+=1
            with log_path.open('a') as f:
                f.write(json.dumps({'task':task,'episode':eid,'error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc()},ensure_ascii=False)+'\n')
        finally:
            if env is not None: env.close()
        (out/'progress.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps({'task':task,'episode':eid,**summary}),flush=True)
    (out/'progress.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


def _sha256(path: Path) -> str:
    import hashlib
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()


if __name__ == '__main__': main()
