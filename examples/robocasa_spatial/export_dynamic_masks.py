"""Export per-window robot/dynamic-object masks from recorded MuJoCo states.

Run in the RoboCasa environment. The mask pixel coordinates are the original
frame-0 Agentview coordinates. Windows with camera motion beyond the configured
threshold are marked invalid; this exporter never silently overlays masks from
misaligned camera frames.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import math
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from examples.robocasa_spatial.generate_teacher_cache import (
    CAMERA_KEYS, camera_calibration, robot_geom_ids, robot_mask,
)

ROOT = Path("/data4/xuchenfei/robocasa365")
AUDIT = ROOT / "output/anchor_visual_audit"
DATA = ROOT / "third_party/robocasa/datasets/v1.0/pretrain/atomic"
GEOMETRY = AUDIT / "geometry_staging"
DEFAULT_MANIFEST = AUDIT / "window_manifest.csv"
DEFAULT_OUTPUT = AUDIT / "dynamic_masks/provisional_v2"
TASKS = ("PickPlaceSinkToCounter", "OpenDrawer", "OpenStandMixerHead", "PickPlaceCounterToCabinet")
CAMERAS = tuple(CAMERA_KEYS)
H = W = 256
MUJOCO_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def _audit_module():
    path = AUDIT / "build_audit.py"
    spec = importlib.util.spec_from_file_location("robocasa_dynamic_mask_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import reset helpers from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _quat_angle_deg(q0: np.ndarray, q: np.ndarray) -> float:
    # Copy: callers pass views into the saved state arrays. In-place
    # normalization here would otherwise corrupt the episode baseline.
    q0 = np.array(q0, dtype=np.float64, copy=True); q = np.array(q, dtype=np.float64, copy=True)
    q0 /= max(np.linalg.norm(q0), 1e-12); q /= max(np.linalg.norm(q), 1e-12)
    dot = float(np.clip(abs(np.dot(q0, q)), -1.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def _camera_delta(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    trans = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    rel = a[:3, :3].T @ b[:3, :3]
    angle = math.degrees(math.acos(float(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))))
    return trans, angle


def _geom_id_image(sim, camera: str) -> np.ndarray:
    seg = np.asarray(sim.render(camera_name=camera, width=W, height=H, segmentation=True))
    if seg.shape != (H, W, 2):
        raise RuntimeError(f"unexpected segmentation shape {camera}: {seg.shape}")
    # MuJoCo offscreen output is bottom-origin; dataset video is top-origin.
    return seg[::-1, :, 1].astype(np.int32, copy=False)


def _robot_body_ids(sim, robot_geoms: np.ndarray) -> set[int]:
    bodies = {int(sim.model.geom_bodyid[g]) for g in robot_geoms}
    changed = True
    while changed:
        changed = False
        for bid in range(1, sim.model.nbody):
            if int(sim.model.body_parentid[bid]) in bodies and bid not in bodies:
                bodies.add(bid); changed = True
    return bodies


def _episode_moved_geoms(sim, states: np.ndarray, robot_bodies: set[int],
                         translation_threshold_m: float, rotation_threshold_deg: float):
    sim.set_state_from_flattened(states[0]); sim.forward()
    p0 = np.asarray(sim.data.body_xpos, dtype=np.float64).copy()
    q0 = np.asarray(sim.data.body_xquat, dtype=np.float64).copy()
    moved: set[int] = set()
    max_translation = np.zeros(sim.model.nbody, dtype=np.float64)
    max_rotation = np.zeros(sim.model.nbody, dtype=np.float64)
    for state in states[1:]:
        sim.set_state_from_flattened(state); sim.forward()
        pos = np.asarray(sim.data.body_xpos, dtype=np.float64)
        quat = np.asarray(sim.data.body_xquat, dtype=np.float64)
        max_translation = np.maximum(max_translation, np.linalg.norm(pos - p0, axis=1))
        for bid in range(1, sim.model.nbody):
            max_rotation[bid] = max(max_rotation[bid], _quat_angle_deg(q0[bid], quat[bid]))
    for bid in range(1, sim.model.nbody):
        if bid in robot_bodies:
            continue
        if (max_translation[bid] > translation_threshold_m or
                max_rotation[bid] > rotation_threshold_deg):
            moved.add(bid)
    moved_geoms = np.asarray([
        gid for gid in range(sim.model.ngeom)
        if int(sim.model.geom_bodyid[gid]) in moved
    ], dtype=np.int32)
    details = []
    for bid in sorted(moved):
        details.append({
            "body_id": bid,
            "body_name": sim.model.body_id2name(bid) or f"body_{bid}",
            "max_translation_m": float(max_translation[bid]),
            "max_rotation_deg": float(max_rotation[bid]),
            "geom_ids": [int(g) for g in range(sim.model.ngeom) if int(sim.model.geom_bodyid[g]) == bid],
        })
    return moved_geoms, details


def _open_env(task: str, episode: int, root: Path, meta: dict, dataset_meta: dict, states: np.ndarray):
    import robocasa  # noqa: F401
    import robosuite as suite

    env_kwargs = dict(dataset_meta["env_args"]["env_kwargs"])
    env_kwargs.update(has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False,
                      camera_names=list(CAMERAS), camera_widths=W, camera_heights=H,
                      camera_depths=False, render_gpu_device_id=-1, seed=0)
    env = suite.make(**env_kwargs)
    try:
        env.set_ep_meta(meta); env.reset()
        xml_path = root / "extras" / f"episode_{episode:06d}" / "model.xml.gz"
        xml_text = gzip.open(xml_path, "rt").read()
        env.reset_from_xml_string(env.edit_model_xml(xml_text))
        env.sim.reset(); env.sim.set_state_from_flattened(states[0]); env.sim.forward()
        return env
    except Exception:
        env.close(); raise


def process_episode(task: str, episode: int, starts: list[int], *, output: Path,
                    camera_translation_tolerance_m: float,
                    camera_rotation_tolerance_deg: float,
                    moved_translation_threshold_m: float,
                    moved_rotation_threshold_deg: float) -> dict:
    root = next((DATA / task).glob("*/lerobot"))
    ep_dir = root / "extras" / f"episode_{episode:06d}"
    meta = json.loads((ep_dir / "ep_meta.json").read_text())
    states = np.load(ep_dir / "states.npz", allow_pickle=False)["states"]
    parquet = root / "data" / f"chunk-{episode//1000:03d}" / f"episode_{episode:06d}.parquet"
    n_frames = len(pd.read_parquet(parquet, columns=["action"]))
    if len(states) != n_frames:
        raise RuntimeError(f"state/video table length mismatch: states={len(states)} parquet={n_frames}")
    if any(t < 0 or t >= len(states) for t in starts):
        raise RuntimeError(f"window frame outside saved simulator states: {task} EP{episode}")

    env = _open_env(task, episode, root,
                    meta, json.loads((root / "extras/dataset_meta.json").read_text()), states)
    try:
        sim = env.sim
        robot_geoms = robot_geom_ids(sim)
        robot_bodies = _robot_body_ids(sim, robot_geoms)
        dynamic_geoms, moved_bodies = _episode_moved_geoms(
            sim, states, robot_bodies, moved_translation_threshold_m, moved_rotation_threshold_deg)

        sim.set_state_from_flattened(states[0]); sim.forward()
        source_t = {cam: camera_calibration(sim, cam, H, W)[1] for cam in CAMERAS}
        source_geom_ids = {cam: _geom_id_image(sim, cam) for cam in CAMERAS}
        source_dynamic = {cam: np.isin(source_geom_ids[cam], dynamic_geoms) for cam in CAMERAS}

        geom_path = GEOMETRY / f"{task}_ep{episode:06d}.npz"
        if not geom_path.is_file():
            raise FileNotFoundError(f"missing verified Anchor geometry: {geom_path}")
        with np.load(geom_path, allow_pickle=False) as geom:
            anchor_xyz = np.asarray(geom["anchor_base_xyz_world"], dtype=np.float64)
            anchor_rpy = np.asarray(geom["anchor_base_rpy_world"], dtype=np.float64)
        audit = _audit_module()
        audit.set_base_pose(env, anchor_xyz[:2], float(anchor_rpy[2]), anchor_rpy)
        anchor_t = {cam: camera_calibration(sim, cam, H, W)[1] for cam in CAMERAS}
        anchor_geom_ids = {cam: _geom_id_image(sim, cam) for cam in CAMERAS}
        anchor_dynamic = {cam: np.isin(anchor_geom_ids[cam], dynamic_geoms) for cam in CAMERAS}

        # Return to the exact episode source state before per-frame exports.
        sim.set_state_from_flattened(states[0]); sim.forward()
        per_frame_masks = np.zeros((len(starts), 2, H, W), dtype=bool)
        per_frame_visible = np.zeros_like(per_frame_masks)
        camera_delta = np.zeros((len(starts), 2, 2), dtype=np.float32)
        camera_stable = np.zeros((len(starts), 2), dtype=bool)
        source_dynamic_stack = np.stack([source_dynamic[c] for c in CAMERAS])
        anchor_dynamic_stack = np.stack([anchor_dynamic[c] for c in CAMERAS])
        for i, frame in enumerate(starts):
            sim.set_state_from_flattened(states[frame]); sim.forward()
            for cidx, cam in enumerate(CAMERAS):
                _, current_t, _ = camera_calibration(sim, cam, H, W)
                dt, dr = _camera_delta(source_t[cam], current_t)
                camera_delta[i, cidx] = (dt, dr)
                camera_stable[i, cidx] = (dt <= camera_translation_tolerance_m and
                                          dr <= camera_rotation_tolerance_deg)
                seg_ids = _geom_id_image(sim, cam)
                per_frame_visible[i, cidx] = seg_ids >= 0
                per_frame_masks[i, cidx] = (np.isin(seg_ids, robot_geoms) |
                                            np.isin(seg_ids, dynamic_geoms) |
                                            (seg_ids < 0))
        metadata = {
            "cache_version": "robocasa-dynamic-mask-v2",
            "task": task, "episode": episode,
            "frame_count": int(len(states)), "window_count": len(starts),
            "frame_index_source": "validated_action_window_start",
            "state_source": "episode extras/states.npz full MuJoCo state",
            "camera_names_order": list(CAMERAS), "resolution_hw": [H, W],
            "mask_true_means": "exclude robot, episode-dynamic geoms, and non-visible/background pixels",
            "moved_body_translation_threshold_m": moved_translation_threshold_m,
            "moved_body_rotation_threshold_deg": moved_rotation_threshold_deg,
            "camera_translation_tolerance_m": camera_translation_tolerance_m,
            "camera_rotation_tolerance_deg": camera_rotation_tolerance_deg,
            "robot_geom_count": int(len(robot_geoms)),
            "dynamic_geom_count": int(len(dynamic_geoms)),
            "dynamic_bodies": moved_bodies,
            "source_camera_pose_world_opencv": {c: source_t[c].tolist() for c in CAMERAS},
            "anchor_camera_pose_world_opencv": {c: anchor_t[c].tolist() for c in CAMERAS},
            "valid_camera_frames": {c: int(camera_stable[:, i].sum()) for i, c in enumerate(CAMERAS)},
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp_output = output.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp_output,
            frame_index=np.asarray(starts, dtype=np.int32),
            dynamic_exclude_masks=np.packbits(per_frame_masks.reshape(len(starts), 2, -1), axis=-1),
            visible_masks=np.packbits(per_frame_visible.reshape(len(starts), 2, -1), axis=-1),
            dynamic_mask_source=np.packbits(source_dynamic_stack.reshape(2, -1), axis=-1),
            dynamic_mask_anchor=np.packbits(anchor_dynamic_stack.reshape(2, -1), axis=-1),
            camera_pose_delta=camera_delta,
            camera_stable=camera_stable,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        tmp_output.replace(output)
        return {"task": task, "episode_index": episode, "status": "complete",
                "windows": len(starts), "dynamic_geom_count": int(len(dynamic_geoms)),
                "stable_camera_windows_left": int(camera_stable[:, 0].sum()),
                "stable_camera_windows_right": int(camera_stable[:, 1].sum()),
                "max_camera_translation_m": float(camera_delta[:, :, 0].max(initial=0)),
                "max_camera_rotation_deg": float(camera_delta[:, :, 1].max(initial=0))}
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--task", choices=TASKS)
    p.add_argument("--limit-episodes", type=int, default=0)
    p.add_argument("--camera-translation-tolerance-m", type=float, default=0.001)
    p.add_argument("--camera-rotation-tolerance-deg", type=float, default=0.1)
    p.add_argument("--moved-translation-threshold-m", type=float, default=0.001)
    p.add_argument("--moved-rotation-threshold-deg", type=float, default=0.25)
    a = p.parse_args()
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    with a.manifest.open(newline="") as f:
        for r in csv.DictReader(f):
            if a.task and r["task"] != a.task:
                continue
            grouped[(r["task"], int(r["episode_index"]))].append(int(r["frame_index"]))
    keys = sorted(grouped)
    if a.limit_episodes:
        keys = keys[:a.limit_episodes]
    a.output_root.mkdir(parents=True, exist_ok=True)
    progress_tag = f"_{a.task}" if a.task else ""
    progress_path = a.output_root / f"progress{progress_tag}.json"
    failures_path = a.output_root / f"failures{progress_tag}.jsonl"
    expected_windows = sum(len(grouped[key]) for key in keys)
    progress = {"status": "running", "selected_episodes": len(keys), "completed_episodes": 0,
                "skipped_episodes": 0, "failed_episodes": 0, "completed_windows": 0,
                "expected_windows": expected_windows, "failed": []}
    for task, episode in keys:
        out = a.output_root / task / f"episode_{episode:06d}_dynamic_masks.npz"
        if out.is_file():
            try:
                with np.load(out, allow_pickle=False) as z:
                    old = json.loads(str(z["metadata"]))
                    expected_frames = np.asarray(grouped[(task, episode)], np.int32)
                    expected_settings = {
                        "moved_body_translation_threshold_m": a.moved_translation_threshold_m,
                        "moved_body_rotation_threshold_deg": a.moved_rotation_threshold_deg,
                        "camera_translation_tolerance_m": a.camera_translation_tolerance_m,
                        "camera_rotation_tolerance_deg": a.camera_rotation_tolerance_deg,
                    }
                    settings_match = all(old.get(k) == v for k, v in expected_settings.items())
                    if (old.get("cache_version") == "robocasa-dynamic-mask-v2" and settings_match
                            and np.array_equal(z["frame_index"], expected_frames)):
                        progress["skipped_episodes"] += 1
                        progress["completed_windows"] += len(expected_frames)
                        continue
            except Exception:
                pass
        try:
            result = process_episode(
                task, episode, sorted(set(grouped[(task, episode)])), output=out,
                camera_translation_tolerance_m=a.camera_translation_tolerance_m,
                camera_rotation_tolerance_deg=a.camera_rotation_tolerance_deg,
                moved_translation_threshold_m=a.moved_translation_threshold_m,
                moved_rotation_threshold_deg=a.moved_rotation_threshold_deg,
            )
            progress["completed_episodes"] += 1
            progress["completed_windows"] += result["windows"]
            progress["last"] = result
        except Exception as exc:
            record = {"task": task, "episode_index": episode, "error": f"{type(exc).__name__}: {exc}",
                      "traceback": traceback.format_exc()}
            progress["failed_episodes"] += 1; progress["failed"].append(record)
            with failures_path.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            progress["last"] = {k: record[k] for k in ("task", "episode_index", "error")}
        progress_path.write_text(json.dumps(progress, indent=2))
        print(json.dumps(progress["last"], ensure_ascii=False), flush=True)
    progress["status"] = "complete" if progress["failed_episodes"] == 0 and progress["completed_windows"] == expected_windows else "needs_review"
    progress_path.write_text(json.dumps(progress, indent=2))
    print(json.dumps(progress, indent=2), flush=True)


if __name__ == "__main__":
    main()
