import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openpi.training.teacher.provenance import source_episode
from scripts.prepare_data import build_views
from scripts.prepare_data import read_jsonl
from scripts.prepare_data import write_jsonl


def test_views_select_categories_and_preserve_normalization_on_rebuild(tmp_path: Path):
    dataset = tmp_path / "train"
    source = dataset / "Task/lerobot"
    meta = source / "meta"
    meta.mkdir(parents=True)
    (dataset / "episode_index_map.csv").write_text(
        "task,filtered_episode_index,type\nTask,0,human\nTask,1,anchor\nTask,2,perturb_multi\nTask,3,anchor\n"
    )
    (dataset / "human_anchor_pairs.csv").write_text(
        "task,human_episode_index,anchor_episode_index,anchor_generation_method\nTask,0,3,human_copy\n"
    )
    data_path = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    video_path = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    info = {
        "features": {"camera": {"dtype": "video"}},
        "chunks_size": 2,
        "data_path": data_path,
        "video_path": video_path,
    }
    (meta / "info.json").write_text(json.dumps(info))
    for name in ["modality.json", "embodiment.json", "tasks.jsonl"]:
        (meta / name).write_text("{}\n")

    def path_for(template, episode):
        return template.format(episode_chunk=episode // 2, episode_index=episode, video_key="camera")

    episodes, stats = [], []
    start = 0
    for episode, size in enumerate([3, 5, 7, 3]):
        parquet = source / path_for(data_path, episode)
        parquet.parent.mkdir(parents=True, exist_ok=True)
        value_id = 0 if episode == 3 else episode
        vector_type = pa.list_(pa.float64()) if episode == 3 else pa.list_(pa.float64(), 2)
        table = pa.table(
            {
                "episode_index": [episode] * size,
                "index": range(start, start + size),
                "frame_index": range(size),
                "timestamp": [i / 20 for i in range(size)],
                "action": pa.array([[value_id, i] for i in range(size)], type=vector_type),
                "observation.state": pa.array([[i, value_id] for i in range(size)], type=vector_type),
            }
        )
        pq.write_table(table, parquet)
        video = source / path_for(video_path, episode)
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"video")
        extras = source / "extras" / f"episode_{episode:06d}"
        extras.mkdir(parents=True)
        (extras / "states.npz").write_bytes(b"states")
        episodes.append({"episode_index": episode, "length": size})
        stats.append(
            {
                "episode_index": episode,
                "stats": {
                    "episode_index": {"mean": [episode], "std": [0], "count": [size]},
                    "index": {"min": [start], "max": [start + size - 1], "mean": [start + (size - 1) / 2]},
                },
            }
        )
        start += size
    write_jsonl(meta / "episodes.jsonl", episodes)
    write_jsonl(meta / "episodes_stats.jsonl", stats)
    (source / "extras/dataset_meta.json").write_text("{}")

    output = tmp_path / "views"
    reference = output / "human/Task/meta/stats.json"
    reference.parent.mkdir(parents=True)
    norm = b'{"state": {"mean": [123], "std": [7]}}\n'
    reference.write_bytes(norm)
    for _ in range(2):
        assert build_views(dataset, output) == {"human": 1, "anchor": 2, "all": 4}
        for view, ids, frames in [("human", [0], 3), ("anchor", [1, 3], 8), ("all", [0, 1, 2, 3], 18)]:
            task = output / view / "Task"
            info = json.loads((task / "meta/info.json").read_text())
            assert info["total_frames"] == frames
            assert info["total_chunks"] == (len(ids) + 1) // 2
            assert info["total_videos"] == len(ids)
            assert (task / "meta/stats.json").read_bytes() == norm
            rows = read_jsonl(task / "meta/episodes.jsonl")
            assert [row["episode_index"] for row in rows] == list(range(len(ids)))
            if view == "all":
                for name in ["data", "videos", "extras"]:
                    assert (task / name).resolve() == source / name
                continue
            assert [row["train_episode_index"] for row in rows] == ids
            assert len(list((task / "data").rglob("*.parquet"))) == len(ids)
            assert len(list((task / "videos").rglob("*.mp4"))) == len(ids)
            start = 0
            for local, original in enumerate(ids):
                assert source_episode(task, local) == (source, original)
                new_path = task / path_for(data_path, local)
                old = pq.read_table(source / path_for(data_path, original))
                new = pq.read_table(new_path)
                assert new_path.is_symlink() == (view == "human")
                assert old["episode_index"][0].as_py() == original
                assert new["episode_index"].to_pylist() == [local] * len(new)
                assert new["index"].to_pylist() == list(range(start, start + len(new)))
                values = ["frame_index", "timestamp", "action", "observation.state"]
                assert new.select(values).to_pydict() == old.select(values).to_pydict()
                ep_stats = read_jsonl(task / "meta/episodes_stats.jsonl")[local]
                assert ep_stats["episode_index"] == local
                assert ep_stats["stats"]["episode_index"] == {"mean": [local], "std": [0], "count": [len(new)]}
                assert ep_stats["stats"]["index"]["min"] == [start]
                video = task / path_for(video_path, local)
                source_video = source / path_for(video_path, original)
                extras = task / "extras" / f"episode_{local:06d}"
                source_extras = source / "extras" / f"episode_{original:06d}"
                if original == 3:
                    assert new.schema.equals(pq.read_schema(source / path_for(data_path, 0)), check_metadata=True)
                    assert not video.samefile(source_video)
                    assert video.read_bytes() == source_video.read_bytes()
                    assert not extras.is_symlink()
                    assert not (extras / "states.npz").samefile(source_extras / "states.npz")
                    assert (extras / "states.npz").read_bytes() == b"states"
                else:
                    assert video.resolve() == source_video
                    assert extras.resolve() == source_extras
                start += len(new)


def test_pooled_stats_preserve_constant_actions():
    from openpi.groot_utils.groot_openpi_dataset import compute_overall_statistics
    from openpi.shared.normalize import NormStats

    weights = np.array([101461, 59532, 112240, 130806]) ** 0.4
    values = np.array([0.0, 1.0, 4.0, 8.0])
    tasks = [{"actions": NormStats(mean=np.array([-1.0, v]), std=np.array([0.0, 2.0]))} for v in values]
    with np.errstate(invalid="raise"):
        result = compute_overall_statistics(tasks, weights)["actions"]
    expected_mean = np.average(values, weights=weights)
    expected_variance = 4.0 + np.average((values - expected_mean) ** 2, weights=weights)
    assert result.std[0] == 0
    np.testing.assert_allclose(result.mean, [-1.0, expected_mean])
    np.testing.assert_allclose(result.std, [0.0, np.sqrt(expected_variance)])
