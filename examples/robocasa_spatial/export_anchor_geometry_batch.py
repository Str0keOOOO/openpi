"""Resumably export full Human/Anchor poses and camera geometry for the 400 reviews."""
from __future__ import annotations

import csv
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path("/data4/xuchenfei/robocasa365")
AUDIT = ROOT / "output/anchor_visual_audit"
PROJECT = ROOT / "third_party/openpi"
GEOMETRY = AUDIT / "geometry_staging"
ROBOCASA_PYTHON = ROOT / "envs/robocasa/bin/python"
WORKERS = max(1, int(os.environ.get("ANCHOR_GEOMETRY_WORKERS", "4")))


def _valid_existing(path: Path, task: str, episode: int) -> bool:
    if not path.is_file():
        return False
    try:
        import numpy as np
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            required = {
                "K_robot0_agentview_left", "K_robot0_agentview_right",
                "T_world_source_robot0_agentview_left", "T_world_source_robot0_agentview_right",
                "T_world_anchor_robot0_agentview_left", "T_world_anchor_robot0_agentview_right",
                "robot_mask_source_robot0_agentview_left", "robot_mask_source_robot0_agentview_right",
                "robot_mask_anchor_robot0_agentview_left", "robot_mask_anchor_robot0_agentview_right",
                "human_base_xyz_world", "human_base_quaternion_xyzw",
                "anchor_base_xyz_world", "anchor_base_rpy_world", "metadata",
            }
            if not required.issubset(data.files):
                return False
        if meta.get("task") != task or int(meta.get("episode", -1)) != episode:
            return False
        pose = meta.get("anchor_pose_reconstruction_check") or {}
        candidate = meta.get("candidate_consistency") or {}
        checks = (
            float(pose.get("xy_error_m", float("inf"))) <= 1e-3,
            abs(float(pose.get("yaw_error_deg", float("inf")))) <= 0.1,
            float(candidate.get("xy_error_m", float("inf"))) <= 1e-3,
            abs(float(candidate.get("yaw_error_deg", float("inf")))) <= 0.1,
        )
        return all(checks)
    except Exception:
        return False


def _export_one(task: str, episode: int, out: Path) -> dict:
    cmd = [str(ROBOCASA_PYTHON), "-m", "examples.robocasa_spatial.export_episode_geometry",
           "--task", task, "--episode", str(episode), "--output", str(out)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    start = time.time()
    result = subprocess.run(cmd, cwd=PROJECT, env=env, text=True, capture_output=True, timeout=300)
    if result.returncode:
        raise RuntimeError(f"returncode={result.returncode}; stderr={result.stderr[-2500:]}; stdout={result.stdout[-1000:]}")
    return {"task": task, "episode": episode, "status": "exported",
            "elapsed_seconds": round(time.time() - start, 2)}


def main():
    source = AUDIT / "anchor_candidates.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open(newline="") as f:
        by_key = {(r["task"], int(r["episode_index"])): r for r in csv.DictReader(f)}
    chosen = sorted(by_key)
    if len(chosen) != 400:
        raise RuntimeError(f"expected 400 unique review episodes; got {len(chosen)}")
    GEOMETRY.mkdir(parents=True, exist_ok=True)
    progress_path = AUDIT / "geometry_batch_progress.json"
    failures_path = AUDIT / "geometry_export_failures.jsonl"
    pending = []
    progress = {"total": len(chosen), "completed": 0, "skipped_existing": 0, "failed": 0,
                "workers": WORKERS, "last": None}
    for task, episode in chosen:
        out = GEOMETRY / f"{task}_ep{episode:06d}.npz"
        if _valid_existing(out, task, episode):
            progress["skipped_existing"] += 1
        else:
            pending.append((task, episode, out))
    progress_path.write_text(json.dumps(progress, indent=2))
    print(json.dumps({"event": "batch_start", "workers": WORKERS,
                      "pending": len(pending), "skipped_existing": progress["skipped_existing"]}), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        future_map = {pool.submit(_export_one, task, episode, out): (task, episode)
                      for task, episode, out in pending}
        for done, future in enumerate(concurrent.futures.as_completed(future_map), 1):
            task, episode = future_map[future]
            try:
                record = future.result()
                progress["completed"] += 1
            except Exception as exc:
                record = {"task": task, "episode": episode, "status": "failed", "error": str(exc)}
                progress["failed"] += 1
                with failures_path.open("a") as f:
                    f.write(json.dumps({"task": task, "episode": episode, "error": str(exc)}) + "\n")
            record["accounted"] = progress["completed"] + progress["skipped_existing"] + progress["failed"]
            record["completion_index"] = done
            progress["last"] = record
            progress_path.write_text(json.dumps(progress, indent=2))
            print(json.dumps(record), flush=True)
    progress["finished"] = True
    progress_path.write_text(json.dumps(progress, indent=2))
    print(json.dumps(progress, indent=2), flush=True)


if __name__ == "__main__":
    main()
