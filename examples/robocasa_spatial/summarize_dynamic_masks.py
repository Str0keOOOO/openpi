"""Aggregate resumable per-task dynamic-mask exports and episode metadata."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/data4/xuchenfei/robocasa365")
AUDIT = ROOT / "output/anchor_visual_audit"
TASKS = ("PickPlaceSinkToCounter", "OpenDrawer", "OpenStandMixerHead", "PickPlaceCounterToCabinet")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=AUDIT / "window_manifest.csv")
    p.add_argument("--mask-root", type=Path, default=AUDIT / "dynamic_masks/provisional_v2")
    p.add_argument("--summary-csv", type=Path)
    a = p.parse_args()
    grouped = defaultdict(set)
    with a.manifest.open(newline="") as f:
        for r in csv.DictReader(f):
            grouped[(r["task"], int(r["episode_index"]))].add(int(r["frame_index"]))
    by_task = {}
    task_progress = {}
    for task in TASKS:
        path = a.mask_root / f"progress_{task}.json"
        if path.is_file():
            task_progress[task] = json.loads(path.read_text())
        else:
            task_progress[task] = {"status": "not_started", "selected_episodes": 0,
                                   "completed_episodes": 0, "skipped_episodes": 0,
                                   "failed_episodes": 0, "completed_windows": 0,
                                   "expected_windows": 0}
        expected_eps = sum(key[0] == task for key in grouped)
        expected_windows = sum(len(frames) for key, frames in grouped.items() if key[0] == task)
        pr = task_progress[task]
        by_task[task] = {
            "status": pr.get("status", "missing"),
            "episodes_expected": expected_eps,
            "episodes_completed_or_cached": int(pr.get("completed_episodes", 0)) + int(pr.get("skipped_episodes", 0)),
            "episodes_failed": int(pr.get("failed_episodes", 0)),
            "windows_expected": expected_windows,
            "windows_completed_or_cached": int(pr.get("completed_windows", 0)),
        }
    rows = []
    missing_files = []
    for task, eid in sorted(grouped):
        path = a.mask_root / task / f"episode_{eid:06d}_dynamic_masks.npz"
        if not path.is_file():
            missing_files.append((task, eid))
            continue
        try:
            import numpy as np
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z["metadata"]))
                counts = z["camera_stable"].sum(axis=0).astype(int).tolist()
                deltas = z["camera_pose_delta"]
                dyn = meta.get("dynamic_bodies", [])
                rows.append({
                    "task": task, "episode_index": eid,
                    "window_count_expected": len(grouped[(task, eid)]),
                    "window_count_exported": int(len(z["frame_index"])),
                    "dynamic_geom_count": int(meta.get("dynamic_geom_count", -1)),
                    "dynamic_body_names": json.dumps([b.get("body_name") for b in dyn], ensure_ascii=False),
                    "left_camera_stable_windows": counts[0],
                    "right_camera_stable_windows": counts[1],
                    "max_camera_translation_m": float(deltas[:, :, 0].max(initial=0)),
                    "max_camera_rotation_deg": float(deltas[:, :, 1].max(initial=0)),
                    "cache_version": meta.get("cache_version"),
                })
        except Exception as e:
            rows.append({"task": task, "episode_index": eid, "window_count_expected": len(grouped[(task, eid)]),
                         "window_count_exported": 0, "error": f"{type(e).__name__}: {e}"})
    status = "complete" if not missing_files and all(x["status"] == "complete" for x in by_task.values()) else "running_or_needs_review"
    summary = {
        "status": status,
        "active_episodes_expected": len(grouped),
        "episode_files_found": len(rows) - sum("error" in r for r in rows),
        "episode_files_missing": len(missing_files),
        "episodes_with_read_errors": sum("error" in r for r in rows),
        "windows_expected": sum(len(v) for v in grouped.values()),
        "by_task": by_task,
        "missing_episode_files": [{"task": t, "episode_index": e} for t, e in missing_files],
    }
    summary_path = a.mask_root / "progress.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    csv_path = a.summary_csv or a.mask_root / "dynamic_mask_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("task", "episode_index", "window_count_expected", "window_count_exported",
               "dynamic_geom_count", "dynamic_body_names", "left_camera_stable_windows",
               "right_camera_stable_windows", "max_camera_translation_m", "max_camera_rotation_deg",
               "cache_version", "error")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
