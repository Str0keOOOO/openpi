"""Helpers for reproducible one-window checks against the real adapter."""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path


def single_window_csv(source: str | Path, task: str, episode: int, frame: int | None = None):
    source = Path(source)
    rows = list(csv.DictReader(source.open(newline="")))
    candidates = [r for r in rows if r.get("task") == task and int(r.get("episode_index", -1)) == int(episode)]
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one window-index row for {task} EP{episode}, got {len(candidates)}")
    row = candidates[0]
    spans = json.loads(row["valid_start_intervals_inclusive"])
    starts = [t for a, b in spans for t in range(int(a), int(b) + 1)]
    if not starts:
        raise ValueError(f"no valid action window for {task} EP{episode}")
    chosen = starts[0] if frame is None else int(frame)
    if chosen not in starts:
        raise ValueError(f"frame {chosen} is not a validated Action Horizon start for {task} EP{episode}")
    row["valid_start_intervals_inclusive"] = json.dumps([[chosen, chosen]])
    temp = tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", prefix="spatial_one_window_", delete=False)
    with temp:
        writer = csv.DictWriter(temp, fieldnames=list(row))
        writer.writeheader(); writer.writerow(row)
    return Path(temp.name), chosen
