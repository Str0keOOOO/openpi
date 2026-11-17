"""Validate exact window coverage and frame-aligned 64D teacher caches."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data4/xuchenfei/robocasa365")
AUDIT = ROOT / "output/anchor_visual_audit"
DEFAULT_MANIFEST = AUDIT / "window_manifest.csv"
DEFAULT_DYNAMIC_ROOT = AUDIT / "dynamic_masks/provisional_v2"
TASKS = ("PickPlaceSinkToCounter", "OpenDrawer", "OpenStandMixerHead", "PickPlaceCounterToCabinet")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--window-root", type=Path)
    p.add_argument("--static-root", type=Path)
    p.add_argument("--dynamic-root", type=Path, default=DEFAULT_DYNAMIC_ROOT)
    p.add_argument("--output-json", type=Path)
    p.add_argument("--output-csv", type=Path)
    a = p.parse_args()
    if a.window_root is None:
        a.window_root = AUDIT / "teacher_cache" / "frame_level"
    if a.static_root is None:
        a.static_root = a.window_root / "static_fields"
    grouped: dict[tuple[str, int], set[int]] = defaultdict(set)
    horizons: dict[tuple[str, int], set[int]] = defaultdict(set)
    with a.manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["task"], int(row["episode_index"]))
            grouped[key].add(int(row["frame_index"]))
            horizons[key].add(int(row["action_horizon"]))
    rows = []
    for (task, eid), expected_set in sorted(grouped.items()):
        expected = np.asarray(sorted(expected_set), dtype=np.int32)
        window_path = a.window_root / task / f"episode_{eid:06d}.npz"
        static_path = a.static_root / task / f"episode_{eid:06d}_static_field.npz"
        dynamic_path = a.dynamic_root / task / f"episode_{eid:06d}_dynamic_masks.npz"
        row = {"task": task, "episode_index": eid, "expected_windows": len(expected),
               "status": "valid", "issues": "", "valid_windows": 0,
               "both_views_valid_windows": 0, "left_valid_windows": 0,
               "right_valid_windows": 0, "left_camera_stable_windows": 0,
               "right_camera_stable_windows": 0, "valid_latent_cells": 0,
               "mean_dynamic_excluded_fraction": None,
               "left_valid_pixels_mean": None, "right_valid_pixels_mean": None}
        issues = []
        try:
            for path in (window_path, static_path, dynamic_path):
                if not path.is_file():
                    issues.append("missing:" + str(path))
            if not issues:
                with np.load(window_path, allow_pickle=False) as z:
                    frames = np.asarray(z["frame_index"], np.int32)
                    latent = np.asarray(z["teacher_latent"], np.float32)
                    valid = np.asarray(z["teacher_valid"], bool)
                    ratio = np.asarray(z["valid_pixel_ratio"], np.float32)
                    camera_valid = np.asarray(z["camera_pose_valid"], bool)
                    metadata = json.loads(str(z["metadata"]))
                    if not np.array_equal(frames, expected): issues.append("frame_index_mismatch")
                    if latent.shape != (len(expected), 64) or valid.shape != latent.shape: issues.append("teacher_shape_mismatch")
                    if ratio.shape != (len(expected), 2) or camera_valid.shape != (len(expected), 2): issues.append("camera_metadata_shape_mismatch")
                    if metadata.get("cache_version") != "robocasa-frame-teacher-v3": issues.append("cache_version_mismatch")
                    if metadata.get("teacher_latent_dim") != 64: issues.append("latent_dim_mismatch")
                    if horizons[(task, eid)] != {50}: issues.append("manifest_action_horizon_mismatch")
                    if int(metadata.get("action_horizon", -1)) != next(iter(horizons[(task, eid)])):
                        issues.append("cache_action_horizon_mismatch")
                    if metadata.get("camera_names_order") != ["robot0_agentview_left", "robot0_agentview_right"]: issues.append("camera_order_mismatch")
                    if not np.isfinite(latent[valid]).all(): issues.append("nonfinite_valid_latent")
                    if np.any(latent[~valid] != 0): issues.append("invalid_latents_not_zeroed")
                    row["valid_windows"] = int(valid.any(axis=1).sum())
                    row["both_views_valid_windows"] = int((valid[:, :32].any(axis=1) & valid[:, 32:].any(axis=1)).sum())
                    row["left_valid_windows"] = int(valid[:, :32].any(axis=1).sum())
                    row["right_valid_windows"] = int(valid[:, 32:].any(axis=1).sum())
                    row["valid_latent_cells"] = int(valid.sum())
                    row["left_valid_pixels_mean"] = float(ratio[:, 0].mean())
                    row["right_valid_pixels_mean"] = float(ratio[:, 1].mean())
                with np.load(static_path, allow_pickle=False) as z:
                    smeta = json.loads(str(z["metadata"]))
                    for key, shape in (("parallax_left", (256, 256, 2)), ("parallax_right", (256, 256, 2)),
                                       ("static_valid_left", (256, 256)), ("static_valid_right", (256, 256))):
                        if key not in z.files or z[key].shape != shape: issues.append(f"static_field_shape:{key}")
                    if smeta.get("cache_version") != "robocasa-static-field-v3": issues.append("static_cache_version_mismatch")
                    for key in ("parallax_left", "parallax_right"):
                        if key in z.files and not np.isfinite(z[key]).all(): issues.append(f"nonfinite:{key}")
                with np.load(dynamic_path, allow_pickle=False) as z:
                    dmeta = json.loads(str(z["metadata"]))
                    if dmeta.get("cache_version") != "robocasa-dynamic-mask-v2": issues.append("dynamic_cache_version_mismatch")
                    if not np.array_equal(z["frame_index"], expected): issues.append("dynamic_frame_index_mismatch")
                    if z["camera_stable"].shape != (len(expected), 2): issues.append("dynamic_camera_stable_shape")
                    if not np.array_equal(camera_valid, z["camera_stable"]): issues.append("camera_validity_disagreement")
                    if z["dynamic_exclude_masks"].shape != (len(expected), 2, (256 * 256 + 7) // 8): issues.append("dynamic_mask_shape")
                    if z["visible_masks"].shape != (len(expected), 2, (256 * 256 + 7) // 8): issues.append("visible_mask_shape")
                    row["left_camera_stable_windows"] = int(z["camera_stable"][:, 0].sum())
                    row["right_camera_stable_windows"] = int(z["camera_stable"][:, 1].sum())
                    excluded = np.unpackbits(z["dynamic_exclude_masks"], axis=-1)[..., :256 * 256]
                    row["mean_dynamic_excluded_fraction"] = float(excluded.mean())
        except Exception as exc:
            issues.append(f"read_error:{type(exc).__name__}:{exc}")
        row["status"] = "valid" if not issues else "invalid"
        row["issues"] = ";".join(issues)
        rows.append(row)

    summary = {
        "status": "complete" if all(r["status"] == "valid" for r in rows) else "incomplete_or_invalid",
        "episodes_expected": len(grouped), "episodes_valid": sum(r["status"] == "valid" for r in rows),
        "episodes_invalid": sum(r["status"] != "valid" for r in rows),
        "windows_expected": sum(r["expected_windows"] for r in rows),
        "windows_with_any_teacher": sum(r["valid_windows"] for r in rows),
        "windows_with_both_views": sum(r["both_views_valid_windows"] for r in rows),
        "windows_with_left_view": sum(r["left_valid_windows"] for r in rows),
        "windows_with_right_view": sum(r["right_valid_windows"] for r in rows),
        "valid_latent_cells": sum(r["valid_latent_cells"] for r in rows),
        "by_task": {},
    }
    for task in TASKS:
        task_rows = [r for r in rows if r["task"] == task]
        summary["by_task"][task] = {
            "episodes": len(task_rows), "valid_episodes": sum(r["status"] == "valid" for r in task_rows),
            "windows": sum(r["expected_windows"] for r in task_rows),
            "windows_with_any_teacher": sum(r["valid_windows"] for r in task_rows),
            "windows_with_both_views": sum(r["both_views_valid_windows"] for r in task_rows),
            "windows_with_left_view": sum(r["left_valid_windows"] for r in task_rows),
            "windows_with_right_view": sum(r["right_valid_windows"] for r in task_rows),
            "valid_latent_cells": sum(r["valid_latent_cells"] for r in task_rows),
        }
    csv_path = a.output_csv or a.window_root / "validation.csv"
    json_path = a.output_json or a.window_root / "validation.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        w.writeheader(); w.writerows(rows)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
