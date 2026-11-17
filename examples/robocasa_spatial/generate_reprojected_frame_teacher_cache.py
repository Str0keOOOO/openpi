"""Reproject SHARP depth geometry at every recorded Agentview pose (CPU only).

The existing episode cache contains SHARP-rendered source/Anchor depth and
alpha. This script back-projects valid source depth pixels into 3D, transforms
them with each window's exact MuJoCo camera pose, forward-splats their
correspondences into the current image, and projects them into the fixed
episode Anchor view. It then applies the existing robot/dynamic/visibility and
depth-occlusion masks before 4x4 masked pooling. No pose-magnitude threshold is
used. Newly disoccluded pixels with no source SHARP geometry remain invalid.

This avoids rerunning SHARP or requiring CUDA while still recomputing the
geometric correspondence from the real per-window camera transform.
"""
from __future__ import annotations

import argparse
import csv
import json
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

from examples.robocasa_spatial.geometry import masked_average_pool
from examples.robocasa_spatial.teacher_cache import CAMERAS

ROOT = Path("/data4/xuchenfei/robocasa365")
AUDIT = ROOT / "output/anchor_visual_audit"
DEFAULT_STATIC_ROOT = AUDIT / "teacher_cache/provisional"
DEFAULT_DYNAMIC_ROOT = AUDIT / "dynamic_masks/provisional_v2"
DEFAULT_POSE_ROOT = AUDIT / "dynamic_masks/camera_poses_v1"
DEFAULT_OUTPUT_ROOT = AUDIT / "teacher_cache/frame_level_reprojected"
TASKS = ("PickPlaceSinkToCounter", "OpenDrawer", "OpenStandMixerHead", "PickPlaceCounterToCabinet")
H = W = 256
POOL_SIZE = 4
REFERENCE_OCCLUSION_TOLERANCE_M = 0.02
CURRENT_ZBUFFER_TOLERANCE_M = 0.0015


def _stamp(path: Path) -> dict:
    st = path.stat()
    return {"path": str(path), "size_bytes": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def _unpack(packed: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8), axis=-1)
    needed = int(np.prod(shape[-2:]))
    if bits.shape[:-1] != shape[:-2] or bits.shape[-1] < needed:
        raise ValueError(f"bad packed mask shape {packed.shape}, expected {shape}")
    return bits[..., :needed].reshape(shape).astype(bool, copy=False)


def _assert_pose(pose: np.ndarray, label: str) -> None:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape[-2:] != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"{label} is not a finite 4x4 transform")
    if not np.allclose(pose[..., 3, :], [0, 0, 0, 1], atol=1e-7, rtol=0):
        raise ValueError(f"{label} has an invalid homogeneous row")
    rotations = pose[..., :3, :3]
    if not np.allclose(np.swapaxes(rotations, -1, -2) @ rotations,
                       np.eye(3), atol=1e-5, rtol=0):
        raise ValueError(f"{label} contains a non-rigid rotation matrix")
    if np.any(np.linalg.det(rotations) < 0.999):
        raise ValueError(f"{label} contains a reflected/invalid rotation matrix")


def _depth_points(depth: np.ndarray, alpha: np.ndarray, k: np.ndarray,
                  source_allowed: np.ndarray) -> np.ndarray:
    """Return source-camera XYZ for valid SHARP depth pixels."""
    depth = np.asarray(depth, dtype=np.float32)
    alpha = np.asarray(alpha, dtype=np.float32)
    k = np.asarray(k, dtype=np.float64)
    if depth.shape != (H, W) or alpha.shape != depth.shape or source_allowed.shape != depth.shape:
        raise ValueError("SHARP depth, alpha and source mask must be matching 256x256 arrays")
    valid = (np.isfinite(depth) & (depth > 0) & np.isfinite(alpha) & (alpha > 0.5)
             & source_allowed)
    indices = np.flatnonzero(valid.reshape(-1))
    if not len(indices):
        return np.zeros((3, 0), dtype=np.float64)
    y, x = np.divmod(indices, W)
    pix = np.stack((x, y, np.ones_like(x)), axis=0).astype(np.float64)
    rays = np.linalg.solve(k, pix)
    points = rays * depth.reshape(-1)[indices][None, :]
    return points


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return transform[:3, :3] @ points + transform[:3, 3:4]


def _project(points: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    proj = np.asarray(k, dtype=np.float64) @ points
    z = points[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = proj[0] / z
        v = proj[1] / z
    return u, v, z


def _anchor_correspondences(points_source: np.ndarray, k: np.ndarray,
                            world_from_source: np.ndarray, world_from_anchor: np.ndarray,
                            anchor_depth: np.ndarray, anchor_alpha: np.ndarray,
                            anchor_allowed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute source-point projections and validity in the fixed Anchor view."""
    source_from_anchor = np.linalg.inv(world_from_anchor) @ world_from_source
    points_anchor = _transform_points(source_from_anchor, points_source)
    u, v, z = _project(points_anchor, k)
    finite = np.isfinite(u) & np.isfinite(v) & np.isfinite(z) & (z > 0)
    ui = np.zeros(len(z), dtype=np.int32)
    vi = np.zeros(len(z), dtype=np.int32)
    ui[finite] = np.rint(u[finite]).astype(np.int32)
    vi[finite] = np.rint(v[finite]).astype(np.int32)
    inside = finite & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    good = np.zeros(len(z), dtype=bool)
    idx = np.flatnonzero(inside)
    if len(idx):
        observed = anchor_depth[vi[idx], ui[idx]]
        good[idx] = (anchor_allowed[vi[idx], ui[idx]]
                     & np.isfinite(observed) & (observed > 0)
                     & (np.abs(z[idx] - observed) <= REFERENCE_OCCLUSION_TOLERANCE_M))
    return u.astype(np.float32), v.astype(np.float32), good


def _splat_window(points_source: np.ndarray, anchor_u: np.ndarray,
                  anchor_v: np.ndarray, anchor_valid: np.ndarray,
                  k: np.ndarray, world_from_source: np.ndarray,
                  world_from_current: np.ndarray, current_allowed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward-splat static SHARP points into the current view with a z-buffer."""
    flow_out = np.zeros((H, W, 2), dtype=np.float32)
    valid_out = np.zeros((H, W), dtype=bool)
    if points_source.shape[1] == 0:
        return flow_out, valid_out

    current_from_source = np.linalg.inv(world_from_current) @ world_from_source
    points_current = _transform_points(current_from_source, points_source)
    u_cur, v_cur, z_cur = _project(points_current, k)
    finite = (np.isfinite(u_cur) & np.isfinite(v_cur) & np.isfinite(z_cur) & (z_cur > 0))
    # Remove matrix-multiply roundoff at the identity pose so a source pixel
    # does not leak tiny bilinear weights into a neighboring bin.
    u_round, v_round = np.rint(u_cur), np.rint(v_cur)
    u_cur = np.where(np.abs(u_cur - u_round) < 1e-4, u_round, u_cur)
    v_cur = np.where(np.abs(v_cur - v_round) < 1e-4, v_round, v_cur)
    in_view = finite & (u_cur >= 0) & (u_cur < W) & (v_cur >= 0) & (v_cur < H)
    src = np.flatnonzero(in_view)
    if not len(src):
        return flow_out, valid_out

    u = u_cur[src].astype(np.float32)
    v = v_cur[src].astype(np.float32)
    z = z_cur[src].astype(np.float32)
    u0, v0 = np.floor(u).astype(np.int32), np.floor(v).astype(np.int32)
    du, dv = u - u0, v - v0
    neighbor_specs = (
        (0, 0, (1 - du) * (1 - dv)),
        (1, 0, du * (1 - dv)),
        (0, 1, (1 - du) * dv),
        (1, 1, du * dv),
    )
    candidates = []
    zbuffer = np.full(H * W, np.inf, dtype=np.float32)
    for ox, oy, weights in neighbor_specs:
        ui, vi = u0 + ox, v0 + oy
        keep = ((weights > 1e-7) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H))
        pids = np.flatnonzero(keep)
        if not len(pids):
            continue
        linear = vi[pids] * W + ui[pids]
        weights_kept = weights[pids].astype(np.float32)
        z_kept = z[pids]
        np.minimum.at(zbuffer, linear, z_kept)
        candidates.append((pids, linear, weights_kept, z_kept, ui[pids], vi[pids]))

    sum_weight = np.zeros(H * W, dtype=np.float64)
    sum_u = np.zeros(H * W, dtype=np.float64)
    sum_v = np.zeros(H * W, dtype=np.float64)
    flow_u = (anchor_u[src] - u_cur[src]).astype(np.float32) / W
    flow_v = (anchor_v[src] - v_cur[src]).astype(np.float32) / H
    for pids, linear, weights, z_kept, ui, vi in candidates:
        front = z_kept <= (zbuffer[linear] + CURRENT_ZBUFFER_TOLERANCE_M)
        on_scene = current_allowed[vi, ui]
        # Anchor validity is indexed by the originating source-depth point.
        point_ids = src[pids]
        use = front & on_scene & anchor_valid[point_ids]
        if not np.any(use):
            continue
        ids = linear[use]
        weighted = weights[use].astype(np.float64)
        np.add.at(sum_weight, ids, weighted)
        np.add.at(sum_u, ids, weighted * flow_u[pids[use]])
        np.add.at(sum_v, ids, weighted * flow_v[pids[use]])

    valid_flat = sum_weight > 1e-10
    flow_flat = flow_out.reshape(-1, 2)
    flow_flat[valid_flat, 0] = (sum_u[valid_flat] / sum_weight[valid_flat]).astype(np.float32)
    flow_flat[valid_flat, 1] = (sum_v[valid_flat] / sum_weight[valid_flat]).astype(np.float32)
    valid_out.reshape(-1)[:] = valid_flat
    return flow_out, valid_out


def _process_episode(task: str, episode: int, frames: list[int], *,
                     static_root: Path, dynamic_root: Path, pose_root: Path,
                     output_root: Path, force: bool) -> dict:
    static_path = static_root / task / f"episode_{episode:06d}.npz"
    diag_path = static_root / task / f"episode_{episode:06d}_diagnostics.npz"
    dynamic_path = dynamic_root / task / f"episode_{episode:06d}_dynamic_masks.npz"
    pose_path = pose_root / task / f"episode_{episode:06d}_camera_poses.npz"
    output_path = output_root / task / f"episode_{episode:06d}.npz"
    for path in (static_path, diag_path, dynamic_path, pose_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    stamps = [_stamp(p) for p in (static_path, diag_path, dynamic_path, pose_path)]
    with np.load(static_path, allow_pickle=False) as z:
        static_meta = json.loads(str(z["metadata"]))
        robot_source = {cam: np.asarray(z[f"robot_mask_current_{cam}"], bool) for cam in CAMERAS}
        robot_anchor = {cam: np.asarray(z[f"robot_mask_reference_{cam}"], bool) for cam in CAMERAS}
    checkpoint_hash = static_meta.get("sharp_checkpoint_sha256")
    if not checkpoint_hash:
        raise ValueError(f"SHARP checkpoint provenance missing in {static_path}")

    if not force and output_path.is_file():
        try:
            with np.load(output_path, allow_pickle=False) as z:
                old_meta = json.loads(str(z["metadata"]))
                if (old_meta.get("cache_version") == "robocasa-frame-teacher-v4-exact-pose-reprojected"
                        and old_meta.get("sharp_checkpoint_sha256") == checkpoint_hash
                        and old_meta.get("input_stamps") == stamps
                        and np.array_equal(z["frame_index"], np.asarray(frames, np.int32))):
                    return {"task": task, "episode_index": episode, "status": "skipped_cache",
                            "windows": len(frames), "valid_windows": int(z["teacher_valid"].any(axis=1).sum())}
        except Exception:
            pass

    with np.load(diag_path, allow_pickle=False) as z:
        depth_source = {cam: np.asarray(z[f"depth_{cam}"][0], np.float32) for cam in CAMERAS}
        alpha_source = {cam: np.asarray(z[f"alpha_{cam}"][0], np.float32) for cam in CAMERAS}
        depth_anchor = {cam: np.asarray(z[f"depth_{cam}"][1], np.float32) for cam in CAMERAS}
        alpha_anchor = {cam: np.asarray(z[f"alpha_{cam}"][1], np.float32) for cam in CAMERAS}
    with np.load(dynamic_path, allow_pickle=False) as z:
        dynamic_meta = json.loads(str(z["metadata"]))
        if dynamic_meta.get("cache_version") != "robocasa-dynamic-mask-v2":
            raise ValueError(f"unsupported dynamic mask version: {dynamic_path}")
        if dynamic_meta.get("camera_names_order") != list(CAMERAS):
            raise ValueError(f"dynamic-mask camera order mismatch: {dynamic_path}")
        if not np.array_equal(np.asarray(z["frame_index"], np.int32), np.asarray(frames, np.int32)):
            raise ValueError(f"dynamic-mask frame indices do not match {task} EP{episode}")
        dynamic_current = _unpack(z["dynamic_exclude_masks"], (len(frames), 2, H, W))
        visible = _unpack(z["visible_masks"], (len(frames), 2, H, W))
        dynamic_source = _unpack(z["dynamic_mask_source"], (2, H, W))
        dynamic_anchor = _unpack(z["dynamic_mask_anchor"], (2, H, W))
    with np.load(pose_path, allow_pickle=False) as z:
        pose_meta = json.loads(str(z["metadata"]))
        if pose_meta.get("cache_version") != "robocasa-window-camera-poses-v1":
            raise ValueError(f"unsupported camera-pose version: {pose_path}")
        if pose_meta.get("camera_names_order") != list(CAMERAS):
            raise ValueError(f"camera-pose order mismatch: {pose_path}")
        if not np.array_equal(np.asarray(z["frame_index"], np.int32), np.asarray(frames, np.int32)):
            raise ValueError(f"camera-pose frame indices do not match {task} EP{episode}")
        current_poses = np.asarray(z["camera_pose_world_opencv"], np.float64)
        source_poses = np.asarray(z["source_camera_pose_world_opencv"], np.float64)
        anchor_poses = np.asarray(z["anchor_camera_pose_world_opencv"], np.float64)
        camera_delta = np.asarray(z["camera_pose_delta_from_source"], np.float32)
    if current_poses.shape != (len(frames), 2, 4, 4):
        raise ValueError(f"unexpected camera pose shape {current_poses.shape}")
    _assert_pose(current_poses, "window camera poses")
    _assert_pose(source_poses, "source camera poses")
    _assert_pose(anchor_poses, "Anchor camera poses")

    latent = np.zeros((len(frames), 64), dtype=np.float32)
    latent_valid = np.zeros((len(frames), 64), dtype=bool)
    valid_pixel_ratio = np.zeros((len(frames), 2), dtype=np.float32)
    valid_bins = np.zeros((len(frames), 2), dtype=np.int8)
    geometry_status = np.full(len(frames), "no_valid_camera", dtype="U32")
    projection = {}
    camera_metadata = {}

    for ci, cam in enumerate(CAMERAS):
        cam_meta = static_meta["cameras"][cam]
        k = np.asarray(cam_meta["intrinsics"], dtype=np.float64)
        world_source = np.asarray(cam_meta["world_from_current_opencv"], dtype=np.float64)
        world_anchor = np.asarray(cam_meta["world_from_anchor_opencv"], dtype=np.float64)
        if max(np.max(np.abs(source_poses[ci] - world_source)),
               np.max(np.abs(anchor_poses[ci] - world_anchor))) > 1e-5:
            raise ValueError(f"camera pose/static SHARP calibration mismatch: {task} EP{episode} {cam}")
        source_allowed = ~(robot_source[cam] | dynamic_source[ci])
        points_source = _depth_points(
            depth_source[cam], alpha_source[cam], k, source_allowed
        )
        if points_source.shape[1] == 0:
            raise ValueError(f"no static SHARP source depth pixels remain for {task} EP{episode} {cam}")
        ref_ok = (np.isfinite(depth_anchor[cam]) & (depth_anchor[cam] > 0)
                  & np.isfinite(alpha_anchor[cam]) & (alpha_anchor[cam] > 0.5)
                  & ~robot_anchor[cam] & ~dynamic_anchor[ci])
        anchor_u, anchor_v, anchor_valid = _anchor_correspondences(
            points_source, k, world_source, world_anchor,
            depth_anchor[cam], alpha_anchor[cam], ref_ok,
        )
        projection[cam] = {
            "k": k, "world_source": world_source, "points_source": points_source,
            "anchor_u": anchor_u, "anchor_v": anchor_v, "anchor_valid": anchor_valid,
        }
        camera_metadata[cam] = {
            "source_static_depth_points": int(points_source.shape[1]),
            "source_static_valid_fraction": float(points_source.shape[1] / (H * W)),
            "anchor_correspondence_valid_fraction": float(anchor_valid.mean()),
            "source_depth_storage_dtype": "float16 on disk, promoted to float32 for projection",
        }

    for i, frame in enumerate(frames):
        views_with_valid_bins = 0
        for ci, cam in enumerate(CAMERAS):
            g = projection[cam]
            current_allowed = visible[i, ci] & ~dynamic_current[i, ci]
            flow, flow_valid = _splat_window(
                g["points_source"], g["anchor_u"], g["anchor_v"],
                g["anchor_valid"], g["k"], g["world_source"], current_poses[i, ci],
                current_allowed,
            )
            pooled, pooled_valid = masked_average_pool(flow, flow_valid, POOL_SIZE)
            latent[i, ci * 32:(ci + 1) * 32] = pooled
            latent_valid[i, ci * 32:(ci + 1) * 32] = pooled_valid
            valid_pixel_ratio[i, ci] = float(flow_valid.mean())
            valid_bins[i, ci] = int(pooled_valid.reshape(16, 2).all(axis=-1).sum())
            views_with_valid_bins += int(pooled_valid.any())
        geometry_status[i] = "valid" if views_with_valid_bins == 2 else (
            "partial_view" if views_with_valid_bins == 1 else "no_valid_camera"
        )

    metadata = {
        "cache_version": "robocasa-frame-teacher-v4-exact-pose-reprojected",
        "task": task, "episode": int(episode), "frame_count": len(frames),
        "teacher_scope": "episode_static_sharp_depth_reprojected_from_each_exact_window_camera_pose_to_reset_anchor",
        "teacher_frame_semantics": "RGB and full MuJoCo camera pose at this window start; SHARP geometry from episode frame 0",
        "camera_names_order": list(CAMERAS),
        "camera_coordinate_convention": "T_world_opencv_camera; x right, y down, z forward",
        "camera_translation_units": "meters",
        "flow_direction": "F(u,v)=(u_anchor-u_current,v_anchor-v_current)",
        "flow_units": "dimensionless pixels normalized by current image width/height",
        "projection_method": "backproject SHARP source depth, transform by exact T_current_from_source, bilinear forward splat with current-view z-buffer, project same points into Anchor",
        "pooling": "masked adaptive average 4x4 per camera; left 32D then right 32D",
        "teacher_latent_dim": 64,
        "pose_magnitude_rejection": "none",
        "reference_occlusion_tolerance_m": REFERENCE_OCCLUSION_TOLERANCE_M,
        "current_zbuffer_tolerance_m": CURRENT_ZBUFFER_TOLERANCE_M,
        "masks": {
            "source": "SHARP source depth/alpha plus source robot and episode-dynamic geom masks",
            "current": "existing per-window robot/dynamic/background geometry-ID masks and visible masks",
            "anchor": "SHARP Anchor depth/alpha plus Anchor robot and episode-dynamic geom masks",
            "projection": "positive finite depth; in-view projection; Anchor depth agreement; current z-buffer",
            "student_rgb": "unmasked three-camera RGB unchanged",
        },
        "camera_geometry": camera_metadata,
        "sharp_version": static_meta.get("sharp_version"),
        "sharp_checkpoint_sha256": checkpoint_hash,
        "calibration_id": static_meta.get("calibration_id"),
        "anchor_source": static_meta.get("anchor_source"),
        "source_camera_poses_world_opencv": {cam: projection[cam]["world_source"].tolist() for cam in CAMERAS},
        "anchor_camera_poses_world_opencv": {cam: np.asarray(static_meta["cameras"][cam]["world_from_anchor_opencv"]).tolist() for cam in CAMERAS},
        "camera_pose_cache_version": pose_meta.get("cache_version"),
        "dynamic_mask_cache_version": dynamic_meta.get("cache_version"),
        "camera_pose_valid_windows": {cam: len(frames) for cam in CAMERAS},
        "camera_pose_delta_max": {
            cam: {"translation_m": float(camera_delta[:, ci, 0].max(initial=0)),
                  "rotation_deg": float(camera_delta[:, ci, 1].max(initial=0))}
            for ci, cam in enumerate(CAMERAS)
        },
        "valid_pixel_ratio_mean": {cam: float(valid_pixel_ratio[:, ci].mean()) for ci, cam in enumerate(CAMERAS)},
        "status_counts": {status: int((geometry_status == status).sum()) for status in np.unique(geometry_status)},
        "input_stamps": stamps,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        frame_index=np.asarray(frames, np.int32),
        teacher_latent=latent,
        teacher_valid=latent_valid,
        valid_pixel_ratio=valid_pixel_ratio,
        valid_pool_bins=valid_bins,
        camera_pose_valid=np.ones((len(frames), 2), dtype=bool),
        camera_pose_world_opencv=current_poses,
        camera_pose_delta=camera_delta,
        geometry_status=geometry_status,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    tmp.replace(output_path)
    return {
        "task": task, "episode_index": episode, "status": "complete", "windows": len(frames),
        "valid_windows": int(latent_valid.any(axis=1).sum()),
        "both_views_valid_windows": int((latent_valid[:, :32].any(axis=1)
                                          & latent_valid[:, 32:].any(axis=1)).sum()),
        "left_valid_windows": int(latent_valid[:, :32].any(axis=1).sum()),
        "right_valid_windows": int(latent_valid[:, 32:].any(axis=1).sum()),
        "left_valid_cells": int(latent_valid[:, :32].sum()),
        "right_valid_cells": int(latent_valid[:, 32:].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=AUDIT / "window_manifest.csv")
    parser.add_argument("--anchor-csv", type=Path, default=AUDIT / "anchor_candidates.csv")
    parser.add_argument("--episodes-csv", type=Path, default=ROOT / "data/training_views/episodes.csv")
    parser.add_argument("--static-root", type=Path, default=DEFAULT_STATIC_ROOT)
    parser.add_argument("--dynamic-root", type=Path, default=DEFAULT_DYNAMIC_ROOT)
    parser.add_argument("--pose-root", type=Path, default=DEFAULT_POSE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--limit-episodes", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with args.episodes_csv.open(newline="") as f:
        retained = {(r["task"], int(r["episode_index"])) for r in csv.DictReader(f)}
    with args.anchor_csv.open(newline="") as f:
        anchor_keys = {(r["task"], int(r["episode_index"])) for r in csv.DictReader(f)}
    if retained - anchor_keys:
        raise ValueError(f"retained episodes missing Anchor rows: {sorted(retained-anchor_keys)}")
    grouped: dict[tuple[str, int], set[int]] = defaultdict(set)
    horizons: dict[tuple[str, int], set[int]] = defaultdict(set)
    with args.manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            key = row["task"], int(row["episode_index"])
            if key not in retained or (args.task and key[0] != args.task):
                continue
            grouped[key].add(int(row["frame_index"]))
            horizons[key].add(int(row["action_horizon"]))
    if any(value != {50} for value in horizons.values()):
        raise ValueError("window manifest must use the configured 50-step action horizon")
    keys = sorted(grouped)
    if args.limit_episodes:
        keys = keys[:args.limit_episodes]
    if not keys:
        raise RuntimeError("no retained episodes/windows selected")

    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "build_progress.json"
    failures_path = args.output_root / "build_failures.jsonl"
    progress = {"status": "running", "selected_episodes": len(keys), "completed_episodes": 0,
                "skipped_episodes": 0, "failed_episodes": 0, "completed_windows": 0,
                "valid_windows": 0, "both_views_valid_windows": 0,
                "left_valid_cells": 0, "right_valid_cells": 0}
    for task, episode in keys:
        try:
            result = _process_episode(
                task, episode, sorted(grouped[(task, episode)]),
                static_root=args.static_root, dynamic_root=args.dynamic_root,
                pose_root=args.pose_root, output_root=args.output_root, force=args.force,
            )
            if result["status"] == "skipped_cache":
                progress["skipped_episodes"] += 1
            else:
                progress["completed_episodes"] += 1
            for key in ("windows", "valid_windows", "both_views_valid_windows", "left_valid_cells", "right_valid_cells"):
                progress["completed_windows" if key == "windows" else key] += result.get(key, 0)
            progress["last"] = result
        except Exception as exc:
            record = {"task": task, "episode_index": episode,
                      "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            progress["failed_episodes"] += 1
            with failures_path.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            progress["last"] = {k: record[k] for k in ("task", "episode_index", "error")}
        progress_path.write_text(json.dumps(progress, indent=2, ensure_ascii=False))
        print(json.dumps(progress["last"], ensure_ascii=False), flush=True)
    progress["status"] = "complete" if not progress["failed_episodes"] else "needs_review"
    progress_path.write_text(json.dumps(progress, indent=2, ensure_ascii=False))
    print(json.dumps(progress, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
