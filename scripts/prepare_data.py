"""Build LeRobot views with local episode indices and shared video files."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openpi.shared.paths import workspace_path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def place_asset(source, dest, *, copy=False):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not copy:
        dest.symlink_to(os.path.relpath(source, dest.parent), target_is_directory=source.is_dir())
    elif source.is_dir():
        shutil.copytree(source, dest)
    else:
        shutil.copy2(source, dest)


def episode_path(info, template, episode, **kwargs):
    return info[template].format(episode_chunk=episode // info["chunks_size"], episode_index=episode, **kwargs)


def write_subset_episode(source, dest, info, video_keys, old_id, new_id, old_start, new_start, paired_human):
    original = source / episode_path(info, "data_path", old_id)
    target = dest / episode_path(info, "data_path", new_id)
    independent = paired_human is not None
    if old_id == new_id and old_start == new_start and not independent:
        place_asset(original, target)
    else:
        table = pq.ParquetFile(original).read(use_threads=False)
        if independent and any(
            not pa.types.is_fixed_size_list(table.schema.field(name).type) for name in ("observation.state", "action")
        ):
            # Older human_copy exports used pandas variable-length lists.
            reference = source / episode_path(info, "data_path", paired_human)
            table = table.cast(pq.read_schema(reference))
        indices = {
            "episode_index": np.full(table.num_rows, new_id, dtype=np.int64),
            "index": np.arange(new_start, new_start + table.num_rows, dtype=np.int64),
        }
        for name, values in indices.items():
            field = table.schema.field(name)
            table = table.set_column(table.schema.get_field_index(name), field, pa.array(values, type=field.type))
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, target, compression="zstd")

    for key in video_keys:
        original = source / episode_path(info, "video_path", old_id, video_key=key)
        target = dest / episode_path(info, "video_path", new_id, video_key=key)
        place_asset(original, target, copy=independent)
    extras = source / "extras" / f"episode_{old_id:06d}"
    if extras.is_dir():
        place_asset(extras, dest / "extras" / f"episode_{new_id:06d}", copy=independent)


def shift_index_stats(row, new_id, frame_shift):
    stats = row["stats"].copy()
    for name, shift in (("episode_index", new_id - row["episode_index"]), ("index", frame_shift)):
        if shift and name in stats:
            # A translation shifts location statistics but leaves std and count intact.
            stats[name] = {
                key: [v + shift for v in values] if key in {"min", "max", "mean"} or key.startswith("q") else values
                for key, values in stats[name].items()
            }
    return {**row, "episode_index": new_id, "stats": stats}


def write_view(source, dest, info, episodes, statistics, starts, normalization, *, subset, human_copies):
    meta = dest / "meta"
    meta.mkdir(parents=True)
    for name in ("modality.json", "embodiment.json", "tasks.jsonl"):
        shutil.copy2(source / "meta" / name, meta / name)
    video_keys = [key for key, feature in info["features"].items() if feature.get("dtype") == "video"]
    if subset:
        extras_meta = source / "extras/dataset_meta.json"
        if extras_meta.is_file():
            place_asset(extras_meta, dest / "extras/dataset_meta.json", copy=bool(human_copies))
    else:
        for name in ("data", "videos", "extras"):
            if (source / name).is_dir():
                place_asset(source / name, dest / name)

    rows, stats = [], []
    start = 0
    for new_id, ep in enumerate(episodes):
        old_id = ep["episode_index"]
        if subset:
            write_subset_episode(
                source, dest, info, video_keys, old_id, new_id, starts[old_id], start, human_copies.get(old_id)
            )
            rows.append({**ep, "episode_index": new_id, "train_episode_index": old_id})
            stats.append(shift_index_stats(statistics[old_id], new_id, start - starts[old_id]))
        else:
            rows.append(ep)
            stats.append(statistics[old_id])
        start += ep["length"]
    count = len(rows)
    view_info = dict(
        info,
        total_episodes=count,
        total_frames=start,
        total_videos=len(video_keys) * count,
        total_chunks=(count + info["chunks_size"] - 1) // info["chunks_size"],
        splits={"train": f"0:{count}"},
    )
    if subset:
        view_info["train_dataset_path"] = os.path.relpath(source, dest)
    (meta / "info.json").write_text(json.dumps(view_info, indent=2) + "\n")
    (meta / "stats.json").write_bytes(normalization)
    write_jsonl(meta / "episodes.jsonl", rows)
    write_jsonl(meta / "episodes_stats.jsonl", stats)


def build_views(dataset, output, normalization_root=None):
    normalization_root = normalization_root or output / "human"
    with (dataset / "episode_index_map.csv").open() as stream:
        categories = {(row["task"], int(row["filtered_episode_index"])): row["type"] for row in csv.DictReader(stream)}
    human_copies = {}
    pairs = dataset / "human_anchor_pairs.csv"
    if pairs.is_file():
        with pairs.open() as stream:
            for row in csv.DictReader(stream):
                if row["anchor_generation_method"] == "human_copy":
                    human_copies.setdefault(row["task"], {})[int(row["anchor_episode_index"])] = int(
                        row["human_episode_index"]
                    )
    sources = sorted(p for p in dataset.iterdir() if (p / "lerobot/meta/info.json").is_file())
    counts = {"human": 0, "anchor": 0, "all": 0}
    output.mkdir(parents=True, exist_ok=True)
    # Build beside the live views so relative links work in both locations.
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staged = Path(temporary)
        for task in sources:
            source = task / "lerobot"
            info = json.loads((source / "meta/info.json").read_text())
            episodes = sorted(read_jsonl(source / "meta/episodes.jsonl"), key=lambda ep: ep["episode_index"])
            statistics = {ep["episode_index"]: ep for ep in read_jsonl(source / "meta/episodes_stats.jsonl")}
            normalization = (normalization_root / task.name / "meta/stats.json").read_bytes()
            starts = {}
            start = 0
            for ep in episodes:
                starts[ep["episode_index"]] = start
                start += ep["length"]
            for view in counts:
                selected = [
                    ep for ep in episodes if view == "all" or categories[task.name, ep["episode_index"]] == view
                ]
                write_view(
                    source,
                    staged / view / task.name,
                    info,
                    selected,
                    statistics,
                    starts,
                    normalization,
                    subset=view != "all",
                    human_copies=human_copies.get(task.name, {}) if view == "anchor" else {},
                )
                counts[view] += len(selected)

        # Publish only the generated tasks; a failed build leaves the live data usable.
        for view in counts:
            (output / view).mkdir(exist_ok=True)
            for task in sources:
                dest = output / view / task.name
                old = staged / "previous" / view / task.name
                if dest.exists():
                    old.parent.mkdir(parents=True, exist_ok=True)
                    dest.rename(old)
                try:
                    (staged / view / task.name).rename(dest)
                except OSError:
                    if old.exists():
                        old.rename(dest)
                    raise
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
