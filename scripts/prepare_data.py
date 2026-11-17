"""Build Human, Anchor and full-data views without copying trajectories."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil

from openpi.shared.paths import workspace_path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def build_views(dataset, output, normalization_root=None):
    normalization_root = normalization_root or output / "human"
    with (dataset / "episode_index_map.csv").open() as stream:
        categories = {(row["task"], int(row["filtered_episode_index"])): row["type"] for row in csv.DictReader(stream)}
    sources = sorted(p for p in dataset.iterdir() if (p / "lerobot/meta/info.json").is_file())
    # Read before writing so existing Human normalization is preserved exactly.
    normalization = {p.name: (normalization_root / p.name / "meta/stats.json").read_bytes() for p in sources}
    counts = {"human": 0, "anchor": 0, "all": 0}
    for task in sources:
        source = task / "lerobot"
        episodes = read_jsonl(source / "meta/episodes.jsonl")
        statistics = read_jsonl(source / "meta/episodes_stats.jsonl")
        for view in counts:
            chosen = [ep for ep in episodes if view == "all" or categories[task.name, ep["episode_index"]] == view]
            ids = {ep["episode_index"] for ep in chosen}
            dest = output / view / task.name
            meta = dest / "meta"
            meta.mkdir(parents=True, exist_ok=True)
            for name in ["data", "videos", "extras"]:
                link = dest / name
                if link.is_symlink():
                    link.unlink()
                elif link.exists():
                    raise FileExistsError(f"{link} contains physical data; expected a link")
                link.symlink_to(os.path.relpath(source / name, dest), target_is_directory=True)
            for name in ["modality.json", "tasks.jsonl", "embodiment.json"]:
                shutil.copy2(source / "meta" / name, meta / name)
            write_jsonl(meta / "episodes.jsonl", chosen)
            write_jsonl(meta / "episodes_stats.jsonl", [row for row in statistics if row["episode_index"] in ids])
            info = json.loads((source / "meta/info.json").read_text())
            cameras = sum(feature.get("dtype") == "video" for feature in info["features"].values())
            info.update(
                total_episodes=len(chosen),
                total_frames=sum(ep["length"] for ep in chosen),
                total_videos=cameras * len(chosen),
                splits={"train": f"0:{len(chosen)}"},
            )
            (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n")
            (meta / "stats.json").write_bytes(normalization[task.name])
            counts[view] += len(chosen)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=workspace_path("data/train"))
    parser.add_argument("--output", type=Path, default=workspace_path("data/views"))
    parser.add_argument("--normalization-root", type=Path, help="Human view containing per-task meta/stats.json")
    args = parser.parse_args()
    print(json.dumps(build_views(args.dataset, args.output, args.normalization_root), indent=2))


if __name__ == "__main__":
    main()
