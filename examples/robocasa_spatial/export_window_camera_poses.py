"""Export exact Agentview poses from LeRobot's recorded base pose.

The two teacher cameras are attached to RoboCasa's ``mobilebase0_support``
body. Their frame-0 camera-to-base transform is fixed; the recorded 16D
LeRobot state contains the base center-site world position and XYZW
orientation. Applying that rigid transform therefore recovers each window's
full camera pose without rebuilding a MuJoCo environment. The result is
checked against the translation/rotation deltas independently exported by the
full MuJoCo dynamic-mask pass.
"""
from __future__ import annotations

import argparse
import csv
import json
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path("/data4/xuchenfei/robocasa365")
AUDIT = ROOT / "output/anchor_visual_audit"
DATA = ROOT / "third_party/robocasa/datasets/v1.0/pretrain/atomic"
CAMERAS = ("robot0_agentview_left", "robot0_agentview_right")
DEFAULT_MANIFEST = AUDIT / "window_manifest.csv"
DEFAULT_DYNAMIC = AUDIT / "dynamic_masks/provisional_v2"
DEFAULT_OUTPUT = AUDIT / "dynamic_masks/camera_poses_v1"


def _stamp(path: Path) -> dict:
    st = path.stat()
    return {"path": str(path), "size_bytes": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def _camera_delta(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    trans = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    rel = a[:3, :3].T @ b[:3, :3]
    angle = float(np.rad2deg(np.arccos(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))))
    return trans, angle


def _base_pose_matrices(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 16:
        raise ValueError(f"expected recorded 16D LeRobot state, got {states.shape}")
    if not np.isfinite(states[:, :7]).all():
        raise ValueError("recorded base pose contains non-finite values")
    transforms = np.repeat(np.eye(4, dtype=np.float64)[None], len(states), axis=0)
    transforms[:, :3, 3] = states[:, :3]
    transforms[:, :3, :3] = Rotation.from_quat(states[:, 3:7]).as_matrix()
    return transforms


def export_episode(task: str, episode: int, frames: list[int], *,
                   dynamic_root: Path, output_root: Path, force: bool) -> dict:
    dynamic_path = dynamic_root / task / f"episode_{episode:06d}_dynamic_masks.npz"
    if not dynamic_path.is_file():
        raise FileNotFoundError(dynamic_path)
    dataset_root = next((DATA / task).glob("*/lerobot"))
    parquet_path = dataset_root / "data" / f"chunk-{episode // 1000:03d}" / f"episode_{episode:06d}.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)
    output_path = output_root / task / f"episode_{episode:06d}_camera_poses.npz"
    stamps = [_stamp(dynamic_path), _stamp(parquet_path)]
    with np.load(dynamic_path, allow_pickle=False) as z:
        dynamic_frames = np.asarray(z["frame_index"], np.int32)
        dynamic_delta = np.asarray(z["camera_pose_delta"], np.float32)
        dynamic_meta = json.loads(str(z["metadata"]))
    expected_frames = np.asarray(frames, np.int32)
    if not np.array_equal(dynamic_frames, expected_frames):
        raise ValueError(f"dynamic-mask frame indices differ for {task} EP{episode}")
    if dynamic_meta.get("cache_version") != "robocasa-dynamic-mask-v2":
        raise ValueError(f"unsupported dynamic-mask cache: {dynamic_path}")
    if dynamic_meta.get("camera_names_order") != list(CAMERAS):
        raise ValueError(f"dynamic-mask camera order mismatch: {dynamic_path}")
    if not force and output_path.is_file():
        try:
            with np.load(output_path, allow_pickle=False) as z:
                old_meta = json.loads(str(z["metadata"]))
                if (old_meta.get("cache_version") == "robocasa-window-camera-poses-v1"
                        and old_meta.get("input_stamps") == stamps
                        and np.array_equal(z["frame_index"], expected_frames)):
                    return {"task": task, "episode_index": episode, "status": "skipped_cache",
                            "windows": len(frames)}
        except Exception:
            pass

    state_column = pd.read_parquet(parquet_path, columns=["observation.state"])["observation.state"]
    states = np.stack(state_column.to_numpy()).astype(np.float64, copy=False)
    if np.any(expected_frames < 0) or np.any(expected_frames >= len(states)):
        raise ValueError(f"window frame outside LeRobot state table for {task} EP{episode}")
    base_world_all = _base_pose_matrices(states)
    base_world = base_world_all[expected_frames]
    base_world_0 = base_world_all[0]

    source_camera = np.stack([
        np.asarray(dynamic_meta["source_camera_pose_world_opencv"][cam], dtype=np.float64)
        for cam in CAMERAS
    ])
    anchor_camera = np.stack([
        np.asarray(dynamic_meta["anchor_camera_pose_world_opencv"][cam], dtype=np.float64)
        for cam in CAMERAS
    ])
    poses = np.empty((len(frames), 2, 4, 4), dtype=np.float64)
    deltas = np.empty((len(frames), 2, 2), dtype=np.float32)
    for ci in range(2):
        # The Agentview camera is rigidly attached to mobilebase0_support.
        base_from_camera = np.linalg.inv(base_world_0) @ source_camera[ci]
        poses[:, ci] = base_world @ base_from_camera
        for i in range(len(frames)):
            deltas[i, ci] = _camera_delta(source_camera[ci], poses[i, ci])
    delta_error = np.abs(deltas - dynamic_delta)
    max_delta_consistency = np.max(delta_error, axis=(0, 1), initial=0)
    # This compares two independently recovered paths: rigid propagation from
    # recorded LeRobot base pose versus camera kinematics in the saved MuJoCo
    # state. It is a consistency assertion, not a motion-magnitude rejection.
    if max_delta_consistency[0] > 5e-5 or max_delta_consistency[1] > 5e-3:
        raise ValueError(
            f"base-to-Agentview rigid-pose assumption disagrees with MuJoCo deltas: "
            f"translation/rotation errors={max_delta_consistency.tolist()}"
        )

    metadata = {
        "cache_version": "robocasa-window-camera-poses-v1",
        "task": task, "episode": int(episode),
        "camera_names_order": list(CAMERAS),
        "frame_index_source": "window_manifest.csv joined to matching LeRobot observation.state rows",
        "state_source": str(parquet_path),
        "state_pose_fields": "observation.state[0:3]=base center-site world xyz; [3:7]=world quaternion XYZW",
        "pose_convention": "T_world_opencv_camera; optical x right, y down, z forward",
        "pose_units": "translation meters, rotation matrix dimensionless",
        "camera_mount": "robot0_agentview_left/right parent_body=mobilebase0_support",
        "recovery": "T_world_base(t) @ inverse(T_world_base(0)) @ T_world_camera(0)",
        "pose_validity": "finite rigid transform and agreement with full MuJoCo camera-delta export; no pose-magnitude threshold",
        "max_delta_disagreement_translation_m_rotation_deg": max_delta_consistency.tolist(),
        "input_stamps": stamps,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        frame_index=expected_frames,
        camera_pose_world_opencv=poses,
        source_camera_pose_world_opencv=source_camera,
        anchor_camera_pose_world_opencv=anchor_camera,
        camera_pose_delta_from_source=deltas,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    tmp.replace(output_path)
    return {
        "task": task, "episode_index": episode, "status": "complete", "windows": len(frames),
        "max_delta_consistency_error_translation_m": float(max_delta_consistency[0]),
        "max_delta_consistency_error_rotation_deg": float(max_delta_consistency[1]),
        "max_translation_delta_m": float(deltas[:, :, 0].max(initial=0)),
        "max_rotation_delta_deg": float(deltas[:, :, 1].max(initial=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dynamic-root", type=Path, default=DEFAULT_DYNAMIC)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task", choices=("PickPlaceSinkToCounter", "OpenDrawer", "OpenStandMixerHead", "PickPlaceCounterToCabinet"))
    parser.add_argument("--limit-episodes", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    grouped: dict[tuple[str, int], set[int]] = defaultdict(set)
    with args.manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            task, episode = row["task"], int(row["episode_index"])
            if args.task and task != args.task:
                continue
            grouped[(task, episode)].add(int(row["frame_index"]))
    keys = sorted(grouped)
    if args.limit_episodes:
        keys = keys[:args.limit_episodes]
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "progress.json"
    failures_path = args.output_root / "failures.jsonl"
    progress = {"status": "running", "selected_episodes": len(keys), "completed_episodes": 0,
                "skipped_episodes": 0, "failed_episodes": 0, "completed_windows": 0}
    for task, episode in keys:
        try:
            result = export_episode(task, episode, sorted(grouped[(task, episode)]),
                                    dynamic_root=args.dynamic_root, output_root=args.output_root,
                                    force=args.force)
            if result["status"] == "skipped_cache":
                progress["skipped_episodes"] += 1
            else:
                progress["completed_episodes"] += 1
            progress["completed_windows"] += result["windows"]
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
