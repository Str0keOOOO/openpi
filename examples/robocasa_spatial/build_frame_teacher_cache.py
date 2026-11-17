"""Legacy fixed-camera pooler; kept only to explain/reproduce old v3 caches.

This CPU-only stage never calls SHARP or MuJoCo. It combines real episode
initial SHARP fields with geometry-ID masks rendered from each saved MuJoCo
state. The retained episode manifest directly determines the training scope.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

from examples.robocasa_spatial.geometry import masked_average_pool
from examples.robocasa_spatial.teacher_cache import CAMERAS

ROOT = Path("/data4/xuchenfei/robocasa365")
AUDIT = ROOT / "output/anchor_visual_audit"
DEFAULT_STATIC_INPUT = AUDIT / "teacher_cache/provisional"
DEFAULT_DYNAMIC_INPUT = AUDIT / "dynamic_masks/provisional_v2"
DEFAULT_MANIFEST = AUDIT / "window_manifest.csv"
H = W = 256
POOL_SIZE = 4


def _unpack(packed: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8), axis=-1)
    prefix = shape[:-2]
    needed = int(shape[-2] * shape[-1])
    if bits.shape[:-1] != prefix or bits.shape[-1] < needed:
        raise ValueError(f"bad packed mask shape {packed.shape}, expected prefix {shape}")
    return bits[..., :needed].reshape(shape).astype(bool, copy=False)


def _source_stamp(path: Path) -> dict:
    st = path.stat()
    return {"path": str(path), "size_bytes": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def _camera_static_valid(flow: np.ndarray, flow_valid: np.ndarray,
                         source_dynamic: np.ndarray, anchor_dynamic: np.ndarray) -> np.ndarray:
    if flow.shape != (H, W, 2) or flow_valid.shape != (H, W):
        raise ValueError(f"unexpected field shapes: {flow.shape}, {flow_valid.shape}")
    if source_dynamic.shape != (H, W) or anchor_dynamic.shape != (H, W):
        raise ValueError("dynamic segmentation must be 256x256")
    if not np.isfinite(flow).all():
        raise ValueError("static parallax field contains non-finite values")
    yy, xx = np.indices((H, W), dtype=np.float32)
    u_ref = xx + flow[..., 0] * W
    v_ref = yy + flow[..., 1] * H
    ui = np.rint(u_ref).astype(np.int32)
    vi = np.rint(v_ref).astype(np.int32)
    in_ref = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ref_dynamic = np.ones((H, W), dtype=bool)
    rows, cols = np.nonzero(in_ref)
    ref_dynamic[rows, cols] = anchor_dynamic[vi[rows, cols], ui[rows, cols]]
    return flow_valid & in_ref & ~source_dynamic & ~ref_dynamic


def _process_episode(task: str, eid: int, frames: list[int], static_root: Path,
                     dynamic_root: Path, static_out_root: Path,
                     window_out_root: Path, force: bool, action_horizon: int) -> dict:
    static_path = static_root / task / f"episode_{eid:06d}.npz"
    diag_path = static_root / task / f"episode_{eid:06d}_diagnostics.npz"
    dynamic_path = dynamic_root / task / f"episode_{eid:06d}_dynamic_masks.npz"
    static_out = static_out_root / task / f"episode_{eid:06d}_static_field.npz"
    window_out = window_out_root / task / f"episode_{eid:06d}.npz"
    for path in (static_path, diag_path, dynamic_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not force and window_out.is_file() and static_out.is_file():
        try:
            with np.load(window_out, allow_pickle=False) as cached:
                meta = json.loads(str(cached["metadata"]))
                if (meta.get("cache_version") == "robocasa-frame-teacher-v3"
                        and int(meta.get("action_horizon", -1)) == action_horizon
                        and np.array_equal(cached["frame_index"], np.asarray(frames, np.int32))
                        and meta.get("input_stamps") == [_source_stamp(p) for p in (static_path, diag_path, dynamic_path)]):
                    return {"task": task, "episode_index": eid, "status": "skipped_cache",
                            "windows": len(frames), "valid_windows": int(cached["teacher_valid"].any(axis=1).sum())}
        except Exception:
            pass

    with np.load(static_path, allow_pickle=False) as cache:
        static_meta = json.loads(str(cache["metadata"]))
        if static_meta.get("cache_version") != "episode-agentview-parallax-v1":
            raise ValueError(f"unsupported static source cache: {static_path}")
        if static_meta.get("task") != task or int(static_meta.get("episode", -1)) != eid:
            raise ValueError(f"static source provenance mismatch: {static_path}")
        flow_valid = {cam: np.asarray(cache[f"flow_valid_{cam}"], bool).copy() for cam in CAMERAS}
        robot_current = {cam: np.asarray(cache[f"robot_mask_current_{cam}"], bool).copy() for cam in CAMERAS}
        robot_reference = {cam: np.asarray(cache[f"robot_mask_reference_{cam}"], bool).copy() for cam in CAMERAS}
    with np.load(diag_path, allow_pickle=False) as diag:
        flows = {cam: np.asarray(diag[f"flow_{cam}"], np.float32).copy() for cam in CAMERAS}
        alphas = {cam: np.asarray(diag[f"alpha_{cam}"], np.float32).copy() for cam in CAMERAS}
    with np.load(dynamic_path, allow_pickle=False) as dynamic:
        dynamic_meta = json.loads(str(dynamic["metadata"]))
        if dynamic_meta.get("cache_version") != "robocasa-dynamic-mask-v2":
            raise ValueError(f"unsupported dynamic mask cache version in {dynamic_path}")
        if dynamic_meta.get("task") != task or int(dynamic_meta.get("episode", -1)) != eid:
            raise ValueError(f"dynamic mask provenance mismatch: {dynamic_path}")
        mask_frames = np.asarray(dynamic["frame_index"], dtype=np.int32)
        if not np.array_equal(mask_frames, np.asarray(frames, np.int32)):
            raise ValueError(f"window index mismatch in {dynamic_path}")
        dynamic_exclude = _unpack(dynamic["dynamic_exclude_masks"], (len(frames), 2, H, W))
        visible = _unpack(dynamic["visible_masks"], (len(frames), 2, H, W))
        source_dynamic = _unpack(dynamic["dynamic_mask_source"], (2, H, W))
        anchor_dynamic = _unpack(dynamic["dynamic_mask_anchor"], (2, H, W))
        stable = np.asarray(dynamic["camera_stable"], bool).copy()
        camera_delta = np.asarray(dynamic["camera_pose_delta"], np.float32).copy()

    if stable.shape != (len(frames), 2):
        raise ValueError(f"camera stability shape mismatch: {stable.shape}")
    static_valid = {}
    for ci, cam in enumerate(CAMERAS):
        if not np.isfinite(alphas[cam]).all() or alphas[cam].shape != (2, H, W):
            raise ValueError(f"invalid alpha diagnostic for {cam}")
        # Recheck the camera-local source/reference robot masks and all static
        # geometry validity. Robot exclusion is already encoded in flow_valid,
        # and is explicitly intersected here for provenance safety.
        base = _camera_static_valid(flows[cam], flow_valid[cam],
                                    source_dynamic[ci] | robot_current[cam],
                                    anchor_dynamic[ci] | robot_reference[cam])
        static_valid[cam] = base

    latent = np.zeros((len(frames), 64), dtype=np.float32)
    latent_valid = np.zeros((len(frames), 64), dtype=bool)
    pixel_ratio = np.zeros((len(frames), 2), dtype=np.float32)
    valid_bins = np.zeros((len(frames), 2), dtype=np.int8)
    camera_pose_valid = stable.copy()
    geometry_status = np.full(len(frames), "no_valid_camera", dtype="U32")
    for i in range(len(frames)):
        used_views = 0
        for ci, cam in enumerate(CAMERAS):
            if not stable[i, ci]:
                continue
            mask = static_valid[cam] & ~dynamic_exclude[i, ci] & visible[i, ci]
            pixel_ratio[i, ci] = float(mask.mean())
            pooled, pooled_valid = masked_average_pool(flows[cam], mask, POOL_SIZE)
            latent[i, ci * 32:(ci + 1) * 32] = pooled
            latent_valid[i, ci * 32:(ci + 1) * 32] = pooled_valid
            valid_bins[i, ci] = int(pooled_valid.reshape(16, 2).all(axis=-1).sum())
            used_views += int(pooled_valid.any())
        geometry_status[i] = "valid" if used_views == 2 else ("partial_view" if used_views == 1 else "no_valid_camera")

    stamps = [_source_stamp(p) for p in (static_path, diag_path, dynamic_path)]
    common_meta = {
        "task": task, "episode": eid,
        "cache_version": "robocasa-frame-teacher-v3",
        "teacher_scope": "episode_static_parallax_field_pooled_with_frame_start_dynamic_mask",
        "camera_names_order": list(CAMERAS),
        "flow_direction": "F(u,v)=(u_anchor-u_current,v_anchor-v_current)",
        "flow_units": "dimensionless_pixels_normalized_by_source_image_width_height",
        "pooling": "masked adaptive average 4x4 per camera; left 32D then right 32D",
        "teacher_latent_dim": 64,
        "action_horizon": int(action_horizon),
        "teacher_frame_semantics": "window start frame; static field from episode frame 0; current robot and episode-dynamic geom IDs excluded",
        "dynamic_mask_cache_version": dynamic_meta.get("cache_version"),
        "camera_stability_thresholds": {
            "translation_m": dynamic_meta.get("camera_translation_tolerance_m"),
            "rotation_deg": dynamic_meta.get("camera_rotation_tolerance_deg"),
        },
        "input_stamps": stamps,
        "anchor_source": static_meta.get("anchor_source"),
        "sharp_checkpoint_sha256": static_meta.get("sharp_checkpoint_sha256"),
        "sharp_version": static_meta.get("sharp_version"),
        "calibration_id": static_meta.get("calibration_id"),
        "human_base_xyz_world": static_meta.get("human_base_xyz_world"),
        "human_base_quaternion_xyzw": static_meta.get("human_base_quaternion_xyzw"),
        "anchor_base_xyz_world": static_meta.get("anchor_base_xyz_world"),
        "anchor_base_rpy_world_rad": static_meta.get("anchor_base_rpy_world_rad"),
        "camera_coordinate_convention": static_meta.get("camera_coordinate_convention"),
        "camera_metadata": static_meta.get("cameras"),
        "rgb_source": static_meta.get("rgb_source"),
        "candidate_consistency": static_meta.get("candidate_consistency"),
    }
    static_meta_out = {**common_meta, "cache_version": "robocasa-static-field-v3",
                       "teacher_scope": "episode_initial_static_parallax_field",
                       "field_shape": [H, W, 2], "validity_shape": [H, W],
                       "static_valid_pixel_ratio": {cam: float(static_valid[cam].mean()) for cam in CAMERAS}}
    static_out.parent.mkdir(parents=True, exist_ok=True)
    tmp_static = static_out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_static,
                        parallax_left=flows[CAMERAS[0]], parallax_right=flows[CAMERAS[1]],
                        static_valid_left=static_valid[CAMERAS[0]], static_valid_right=static_valid[CAMERAS[1]],
                        dynamic_mask_source_left=source_dynamic[0], dynamic_mask_source_right=source_dynamic[1],
                        dynamic_mask_anchor_left=anchor_dynamic[0], dynamic_mask_anchor_right=anchor_dynamic[1],
                        metadata=np.asarray(json.dumps(static_meta_out, sort_keys=True)))
    tmp_static.replace(static_out)

    window_meta = {**common_meta,
                   "source_static_field": str(static_out),
                   "status_counts": {s: int((geometry_status == s).sum()) for s in np.unique(geometry_status)},
                   "valid_pixel_ratio_mean": {cam: float(pixel_ratio[:, ci].mean()) for ci, cam in enumerate(CAMERAS)},
                   "camera_stable_window_count": {cam: int(stable[:, ci].sum()) for ci, cam in enumerate(CAMERAS)}}
    window_out.parent.mkdir(parents=True, exist_ok=True)
    tmp_window = window_out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_window, frame_index=np.asarray(frames, np.int32),
                        teacher_latent=latent, teacher_valid=latent_valid,
                        valid_pixel_ratio=pixel_ratio, valid_pool_bins=valid_bins,
                        camera_pose_valid=camera_pose_valid, camera_pose_delta=camera_delta,
                        geometry_status=geometry_status,
                        metadata=np.asarray(json.dumps(window_meta, sort_keys=True)))
    tmp_window.replace(window_out)
    return {"task": task, "episode_index": eid, "status": "complete", "windows": len(frames),
            "valid_windows": int(latent_valid.any(axis=1).sum()),
            "both_views_valid_windows": int((latent_valid[:, :32].any(axis=1) & latent_valid[:, 32:].any(axis=1)).sum()),
            "left_stable": int(stable[:, 0].sum()), "right_stable": int(stable[:, 1].sum()),
            "left_static_coverage": float(static_valid[CAMERAS[0]].mean()),
            "right_static_coverage": float(static_valid[CAMERAS[1]].mean())}


def main() -> None:
    raise SystemExit(
        "This legacy builder reuses frame-0 pixel flow and rejects camera motion. "
        "For spatially aligned labels use export_window_camera_poses followed by "
        "generate_reprojected_frame_teacher_cache."
    )
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--static-input-root", type=Path, default=DEFAULT_STATIC_INPUT)
    p.add_argument("--dynamic-input-root", type=Path, default=DEFAULT_DYNAMIC_INPUT)
    p.add_argument("--static-output-root", type=Path)
    p.add_argument("--window-output-root", type=Path)
    p.add_argument("--limit-episodes", type=int, default=0)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.window_output_root is None:
        a.window_output_root = AUDIT / "teacher_cache" / "frame_level"
    if a.static_output_root is None:
        a.static_output_root = a.window_output_root / "static_fields"
    grouped: dict[tuple[str, int], set[int]] = defaultdict(set)
    horizons: dict[tuple[str, int], set[int]] = defaultdict(set)
    with a.manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["task"], int(row["episode_index"]))
            grouped[key].add(int(row["frame_index"]))
            if "action_horizon" not in row or not str(row["action_horizon"]).strip():
                raise ValueError("window manifest must include the validated action_horizon column")
            horizons[key].add(int(row["action_horizon"]))
    if any(len(values) != 1 for values in horizons.values()):
        raise ValueError("an episode's manifest windows disagree on Action Horizon")
    keys = sorted(grouped)
    if a.limit_episodes:
        keys = keys[:a.limit_episodes]
    progress_path = a.window_output_root / "build_progress.json"
    failures_path = a.window_output_root / "build_failures.jsonl"
    a.window_output_root.mkdir(parents=True, exist_ok=True)
    progress = {"status": "running", "selected_episodes": len(keys), "completed_episodes": 0,
                "skipped_episodes": 0, "failed_episodes": 0, "completed_windows": 0,
                "valid_windows": 0, "both_views_valid_windows": 0}
    for task, eid in keys:
        try:
            result = _process_episode(task, eid, sorted(grouped[(task, eid)]),
                                      a.static_input_root, a.dynamic_input_root,
                                      a.static_output_root, a.window_output_root, a.force,
                                      next(iter(horizons[(task, eid)])))
            progress["skipped_episodes" if result["status"] == "skipped_cache" else "completed_episodes"] += 1
            progress["completed_windows"] += result["windows"]
            progress["valid_windows"] += result.get("valid_windows", 0)
            progress["both_views_valid_windows"] += result.get("both_views_valid_windows", 0)
            progress["last"] = result
        except Exception as exc:
            record = {"task": task, "episode_index": eid, "error": f"{type(exc).__name__}: {exc}",
                      "traceback": traceback.format_exc()}
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
