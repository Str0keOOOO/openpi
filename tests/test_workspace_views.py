import json
from pathlib import Path

from scripts.prepare_data import build_views


def test_views_select_categories_and_preserve_normalization_on_rebuild(tmp_path: Path):
    dataset = tmp_path / "train"
    source = dataset / "Task/lerobot"
    meta = source / "meta"
    meta.mkdir(parents=True)
    for name in ["data", "videos", "extras"]:
        (source / name).mkdir()
    (dataset / "episode_index_map.csv").write_text(
        "task,filtered_episode_index,type\nTask,0,human\nTask,1,anchor\nTask,2,perturb_multi\n"
    )
    for name in ["modality.json", "embodiment.json", "tasks.jsonl"]:
        (meta / name).write_text("{}\n")
    rows = [{"episode_index": i, "length": n} for i, n in enumerate([3, 5, 7])]
    for name in ["episodes.jsonl", "episodes_stats.jsonl"]:
        (meta / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    (meta / "info.json").write_text(json.dumps({"features": {"camera": {"dtype": "video"}}}))
    output = tmp_path / "views"
    reference = output / "human/Task/meta/stats.json"
    reference.parent.mkdir(parents=True)
    norm = b'{"state": {"mean": [123], "std": [7]}}\n'
    reference.write_bytes(norm)
    for _ in range(2):
        assert build_views(dataset, output) == {"human": 1, "anchor": 1, "all": 3}
        for view, ids, frames in [("human", [0], 3), ("anchor", [1], 5), ("all", [0, 1, 2], 15)]:
            path = output / view / "Task"
            actual = [
                json.loads(line)["episode_index"] for line in (path / "meta/episodes.jsonl").read_text().splitlines()
            ]
            assert actual == ids
            assert json.loads((path / "meta/info.json").read_text())["total_frames"] == frames
            assert (path / "meta/stats.json").read_bytes() == norm
            for name in ["data", "videos", "extras"]:
                assert (path / name).is_symlink()
                assert (path / name).resolve() == source / name
