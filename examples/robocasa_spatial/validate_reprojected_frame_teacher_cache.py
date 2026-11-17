"""Validate frame teachers generated from exact window camera poses."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data4/xuchenfei/robocasa365")
AUDIT = ROOT / "output/anchor_visual_audit"
TASKS = ("PickPlaceSinkToCounter", "OpenDrawer", "OpenStandMixerHead", "PickPlaceCounterToCabinet")
EXPECTED_VERSION = "robocasa-frame-teacher-v4-exact-pose-reprojected"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=AUDIT / "window_manifest.csv")
    parser.add_argument("--window-root", type=Path, default=AUDIT / "teacher_cache/frame_level")
    parser.add_argument("--dynamic-root", type=Path, default=AUDIT / "dynamic_masks/provisional_v2")
    parser.add_argument("--pose-root", type=Path, default=AUDIT / "dynamic_masks/camera_poses_v1")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()

    grouped: dict[tuple[str, int], set[int]] = defaultdict(set)
    with args.manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            grouped[(row["task"], int(row["episode_index"]))].add(int(row["frame_index"]))

    rows = []
    for (task, episode), expected_set in sorted(grouped.items()):
        expected = np.asarray(sorted(expected_set), np.int32)
        window_path = args.window_root / task / f"episode_{episode:06d}.npz"
        dynamic_path = args.dynamic_root / task / f"episode_{episode:06d}_dynamic_masks.npz"
        pose_path = args.pose_root / task / f"episode_{episode:06d}_camera_poses.npz"
        row = {"task": task, "episode_index": episode, "expected_windows": len(expected),
               "valid_windows": 0, "both_views_valid_windows": 0, "left_valid_windows": 0,
               "right_valid_windows": 0, "valid_latent_cells": 0,
               "pose_matrix_max_abs_mismatch": None, "pose_delta_max_abs_mismatch": None,
               "status": "valid", "issues": ""}
        issues = []
        try:
            for path in (window_path, dynamic_path, pose_path):
                if not path.is_file():
                    issues.append("missing:" + str(path))
            if not issues:
                with np.load(window_path, allow_pickle=False) as cache:
                    frames = np.asarray(cache["frame_index"], np.int32)
                    latent = np.asarray(cache["teacher_latent"], np.float32)
                    valid = np.asarray(cache["teacher_valid"], bool)
                    poses = np.asarray(cache["camera_pose_world_opencv"], np.float64)
                    pose_valid = np.asarray(cache["camera_pose_valid"], bool)
                    metadata = json.loads(str(cache["metadata"]))
                    if metadata.get("cache_version") != EXPECTED_VERSION:
                        issues.append("cache_version_mismatch")
                    if not np.array_equal(frames, expected):
                        issues.append("window_frame_index_mismatch")
                    if latent.shape != (len(expected), 64) or valid.shape != latent.shape:
                        issues.append("teacher_shape_mismatch")
                    if poses.shape != (len(expected), 2, 4, 4) or pose_valid.shape != (len(expected), 2):
                        issues.append("pose_shape_mismatch")
                    if np.any(~pose_valid):
                        issues.append("invalid_camera_pose_flag")
                    if not np.isfinite(poses).all():
                        issues.append("nonfinite_camera_pose")
                    if not np.isfinite(latent[valid]).all():
                        issues.append("nonfinite_valid_teacher")
                    if np.any(latent[~valid] != 0):
                        issues.append("invalid_teacher_cells_not_zeroed")
                    if metadata.get("camera_names_order") != [
                        "robot0_agentview_left", "robot0_agentview_right"
                    ]:
                        issues.append("camera_order_mismatch")
                    if "camera_stability_thresholds" in metadata:
                        issues.append("legacy_camera_threshold_still_recorded")
                    row["valid_windows"] = int(valid.any(axis=1).sum())
                    row["both_views_valid_windows"] = int(
                        (valid[:, :32].any(axis=1) & valid[:, 32:].any(axis=1)).sum()
                    )
                    row["left_valid_windows"] = int(valid[:, :32].any(axis=1).sum())
                    row["right_valid_windows"] = int(valid[:, 32:].any(axis=1).sum())
                    row["valid_latent_cells"] = int(valid.sum())

                with np.load(pose_path, allow_pickle=False) as pose_cache:
                    pose_frames = np.asarray(pose_cache["frame_index"], np.int32)
                    exported_poses = np.asarray(pose_cache["camera_pose_world_opencv"], np.float64)
                    pose_deltas = np.asarray(pose_cache["camera_pose_delta_from_source"], np.float32)
                    if not np.array_equal(pose_frames, expected):
                        issues.append("pose_frame_index_mismatch")
                with np.load(dynamic_path, allow_pickle=False) as dynamic_cache:
                    dynamic_frames = np.asarray(dynamic_cache["frame_index"], np.int32)
                    dynamic_deltas = np.asarray(dynamic_cache["camera_pose_delta"], np.float32)
                    if not np.array_equal(dynamic_frames, expected):
                        issues.append("dynamic_frame_index_mismatch")
                if poses.shape == exported_poses.shape:
                    pose_mismatch = float(np.max(np.abs(poses - exported_poses), initial=0))
                    row["pose_matrix_max_abs_mismatch"] = pose_mismatch
                    if pose_mismatch > 1e-10:
                        issues.append("pose_matrix_cache_mismatch")
                if pose_deltas.shape == dynamic_deltas.shape:
                    delta_mismatch = float(np.max(np.abs(pose_deltas - dynamic_deltas), initial=0))
                    row["pose_delta_max_abs_mismatch"] = delta_mismatch
                    if delta_mismatch > 1e-4:
                        issues.append("pose_delta_vs_mask_export_mismatch")
                else:
                    issues.append("pose_delta_shape_mismatch")
        except Exception as exc:
            issues.append(f"read_error:{type(exc).__name__}:{exc}")
        row["status"] = "valid" if not issues else "invalid"
        row["issues"] = ";".join(issues)
        rows.append(row)

    summary = {
        "status": "complete" if all(row["status"] == "valid" for row in rows) else "incomplete_or_invalid",
        "episodes_expected": len(grouped),
        "episodes_valid": sum(row["status"] == "valid" for row in rows),
        "episodes_invalid": sum(row["status"] != "valid" for row in rows),
        "windows_expected": sum(row["expected_windows"] for row in rows),
        "windows_with_any_teacher": sum(row["valid_windows"] for row in rows),
        "windows_with_both_views": sum(row["both_views_valid_windows"] for row in rows),
        "windows_with_left_view": sum(row["left_valid_windows"] for row in rows),
        "windows_with_right_view": sum(row["right_valid_windows"] for row in rows),
        "valid_latent_cells": sum(row["valid_latent_cells"] for row in rows),
        "by_task": {},
    }
    for task in TASKS:
        subset = [row for row in rows if row["task"] == task]
        summary["by_task"][task] = {
            "episodes": len(subset),
            "valid_episodes": sum(row["status"] == "valid" for row in subset),
            "windows": sum(row["expected_windows"] for row in subset),
            "windows_with_any_teacher": sum(row["valid_windows"] for row in subset),
            "windows_with_both_views": sum(row["both_views_valid_windows"] for row in subset),
        }
    csv_path = args.output_csv or args.window_root / "validation_reprojected.csv"
    json_path = args.output_json or args.window_root / "validation_reprojected.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
