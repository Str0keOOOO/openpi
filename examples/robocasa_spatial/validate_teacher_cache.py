"""Audit a generated provisional/formal episode-level teacher cache."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("/data4/xuchenfei/robocasa365")
AUDIT = ROOT / "output/anchor_visual_audit"
DEFAULT_EXCLUSIONS = AUDIT / "manual_episode_exclusions.csv"
TASKS = (
    "PickPlaceSinkToCounter",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
)
CAMERAS = ("robot0_agentview_left", "robot0_agentview_right")


def _read_expected(anchor_csv: Path, motion_csv: Path, exclusions_csv: Path
                   ) -> tuple[list[dict], set[tuple[str, int]], set[tuple[str, int]]]:
    rows = list(csv.DictReader(anchor_csv.open(newline="")))
    candidates = {(r["task"], int(r["episode_index"])): r for r in rows}
    if len(candidates) != 400:
        raise RuntimeError(f"Expected 400 unique candidates; got {len(candidates)}")
    mobile: set[tuple[str, int]] = set()
    for r in csv.DictReader(motion_csv.open(newline="")):
        if r["task"] not in TASKS:
            continue
        if float(r["max_xy_m"]) > 0.01 or float(r["max_yaw_deg"]) > 1.0:
            mobile.add((r["task"], int(r["episode_index"])))
    manually_excluded: set[tuple[str, int]] = set()
    if exclusions_csv.is_file():
        for r in csv.DictReader(exclusions_csv.open(newline="")):
            if str(r.get("exclude_from_training", "")).strip().lower() in ("1", "true", "yes"):
                manually_excluded.add((r["task"], int(r["episode_index"])))
    expected = [r for key, r in candidates.items()
                if r.get("status") == "rendered_candidate" and key not in mobile
                and key not in manually_excluded]
    return expected, mobile, manually_excluded


def validate_one(cache: Path, diagnostic: Path, expected_key: tuple[str, int], mode: str) -> dict:
    task, episode = expected_key
    row: dict = {"task": task, "episode_index": episode, "cache_path": str(cache),
                 "diagnostics_path": str(diagnostic), "status": "valid", "issues": ""}
    issues: list[str] = []
    try:
        with np.load(cache, allow_pickle=False) as z:
            required = {"latent", "valid", "metadata"}
            required |= {f"{prefix}_{camera}" for camera in CAMERAS
                         for prefix in ("robot_mask_current", "robot_mask_reference", "flow_valid")}
            absent = sorted(required - set(z.files))
            if absent:
                issues.append("missing_cache_keys:" + ",".join(absent))
            latent = np.asarray(z["latent"], dtype=np.float32)
            valid = np.asarray(z["valid"], dtype=bool)
            metadata = json.loads(str(z["metadata"]))
            if latent.shape != (64,) or valid.shape != (64,):
                issues.append(f"bad_latent_shapes:{latent.shape}/{valid.shape}")
            if not np.isfinite(latent).all():
                issues.append("nonfinite_latent")
            valid_cells = int(valid.sum())
            row["valid_cells"] = valid_cells
            if valid_cells == 0:
                issues.append("no_valid_latent_cells")
            if metadata.get("task") != task or int(metadata.get("episode", -1)) != episode:
                issues.append("episode_provenance_mismatch")
            if metadata.get("mode") != mode:
                issues.append(f"mode_mismatch:{metadata.get('mode')}")
            if metadata.get("anchor_review_status") != "unreviewed_provisional" and mode == "provisional":
                issues.append("provisional_anchor_status_missing")
            if metadata.get("teacher_scope") != "episode_initial_static_scene":
                issues.append("teacher_scope_mismatch")
            cameras = metadata.get("cameras", {})
            for i, camera in enumerate(CAMERAS):
                cm = cameras.get(camera)
                if cm is None:
                    issues.append(f"camera_metadata_missing:{camera}")
                    continue
                cell_count = int(cm.get("latent_valid_cells", -1))
                row[f"{camera}_valid_cells"] = cell_count
                row[f"{camera}_flow_valid_fraction"] = cm.get("valid_flow_fraction")
                row[f"{camera}_robot_mask_fraction_current"] = cm.get("robot_mask_fraction_current")
                if cell_count != int(valid[i * 32:(i + 1) * 32].sum()):
                    issues.append(f"camera_mask_count_mismatch:{camera}")
                for key in (f"robot_mask_current_{camera}", f"robot_mask_reference_{camera}",
                            f"flow_valid_{camera}"):
                    if key in z.files and z[key].shape != (256, 256):
                        issues.append(f"bad_pixel_mask_shape:{key}:{z[key].shape}")
            if not Path(diagnostic).is_file():
                issues.append("missing_depth_flow_diagnostics")
    except Exception as exc:
        issues.append(f"cache_read_error:{type(exc).__name__}:{exc}")

    if not issues and diagnostic.is_file():
        try:
            with np.load(diagnostic, allow_pickle=False) as d:
                for camera in CAMERAS:
                    flow_name = f"flow_{camera}"
                    if flow_name not in d.files:
                        issues.append(f"diagnostic_missing:{flow_name}")
                    elif d[flow_name].shape != (256, 256, 2):
                        issues.append(f"diagnostic_shape:{flow_name}:{d[flow_name].shape}")
                    for prefix in ("depth", "alpha"):
                        name = f"{prefix}_{camera}"
                        if name not in d.files:
                            issues.append(f"diagnostic_missing:{name}")
                        elif d[name].shape != (2, 256, 256):
                            issues.append(f"diagnostic_shape:{name}:{d[name].shape}")
        except Exception as exc:
            issues.append(f"diagnostic_read_error:{type(exc).__name__}:{exc}")
    row["status"] = "valid" if not issues else "invalid"
    row["issues"] = ";".join(issues)
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", type=Path, default=AUDIT / "teacher_cache/provisional")
    p.add_argument("--anchor-csv", type=Path, default=AUDIT / "anchor_candidates.csv")
    p.add_argument("--motion-csv", type=Path, default=ROOT / "output/base_motion_audit/episodes.csv")
    p.add_argument("--exclusions-csv", type=Path, default=DEFAULT_EXCLUSIONS)
    p.add_argument("--mode", choices=("provisional", "formal"), default="provisional")
    p.add_argument("--output-json", type=Path)
    p.add_argument("--output-csv", type=Path)
    a = p.parse_args()

    expected, mobile, manually_excluded = _read_expected(a.anchor_csv, a.motion_csv, a.exclusions_csv)
    rows = []
    for r in sorted(expected, key=lambda x: (x["task"], int(x["episode_index"]))):
        task, episode = r["task"], int(r["episode_index"])
        cache = a.cache_root / task / f"episode_{episode:06d}.npz"
        diagnostic = a.cache_root / task / f"episode_{episode:06d}_diagnostics.npz"
        rows.append(validate_one(cache, diagnostic, (task, episode), a.mode))

    by_task: dict[str, dict] = {}
    for task in TASKS:
        group = [r for r in rows if r["task"] == task]
        by_task[task] = {
            "expected": len(group),
            "valid": sum(r["status"] == "valid" for r in group),
            "invalid": sum(r["status"] != "valid" for r in group),
            "mean_valid_cells_total": float(np.mean([r.get("valid_cells", 0) for r in group])) if group else None,
            "mean_left_valid_cells": float(np.mean([r.get(f"{CAMERAS[0]}_valid_cells", 0) for r in group])) if group else None,
            "mean_right_valid_cells": float(np.mean([r.get(f"{CAMERAS[1]}_valid_cells", 0) for r in group])) if group else None,
        }
    issue_counts = Counter(issue.split(":", 1)[0]
                           for r in rows for issue in r["issues"].split(";") if issue)
    summary = {
        "mode": a.mode,
        "expected_static_episodes": len(expected),
        "mobile_episodes_excluded": len(mobile),
        "manual_episode_exclusions": len(manually_excluded),
        "excluded_cache_files_retained_but_ignored": sum(
            (a.cache_root / task / f"episode_{episode:06d}.npz").is_file()
            for task, episode in manually_excluded
        ),
        "validated_caches": sum(r["status"] == "valid" for r in rows),
        "invalid_caches": sum(r["status"] != "valid" for r in rows),
        "by_task": by_task,
        "issue_counts": dict(issue_counts),
        "status": "verified" if all(r["status"] == "valid" for r in rows) else "needs_review",
        "note": "A cache passing file/numeric validation is not an anchor approval; provisional stays excluded from formal training.",
    }
    out_json = a.output_json or a.cache_root / "validation.json"
    out_csv = a.output_csv or a.cache_root / "validation.csv"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(json.dumps(summary, indent=2))
    if summary["status"] != "verified":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
