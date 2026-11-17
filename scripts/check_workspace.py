"""Check dataset views and optionally run a checkpoint on a real observation."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from openpi.shared.paths import workspace_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--policy-config", default="pi05_robocasa_human418_baseline")
    parser.add_argument("--observation", type=Path, help="NPZ captured by check_sim.py")
    parser.add_argument("--save-actions", type=Path)
    args = parser.parse_args()
    import jax
    import numpy as np

    from openpi.groot_utils.groot_openpi_dataset import GrootOpenpiSingleDataset
    from openpi.training import config

    with workspace_path("data/train/episode_index_map.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    counts = Counter(row["type"] for row in rows)
    expected = {"human": counts["human"], "anchor": counts["anchor"], "all": len(rows)}
    for view, count in expected.items():
        actual = 0
        for path in workspace_path("data/views", view).glob("*/meta/info.json"):
            actual += json.loads(path.read_text())["total_episodes"]
            for name in ["data", "videos", "extras"]:
                assert (path.parent.parent / name).is_dir(), path.parent.parent / name
        assert actual == count, (view, actual, count)
    # Read a real native Groot item, including all three video streams.
    train = config.get_config("pi05_robocasa_human418_baseline")
    sample = GrootOpenpiSingleDataset(train.data.data_dirs[0], train.model.action_horizon)[0]
    assert sample["actions"].shape == (50, 12)
    assert sample["observation/state"].shape == (16,)
    for key in ["observation/image", "observation/right_image", "observation/wrist_image"]:
        assert np.isfinite(sample[key]).all()
    report = {
        "views": expected,
        "sample_actions": list(sample["actions"].shape),
        "jax_devices": [str(d) for d in jax.devices()],
    }
    if args.checkpoint:
        from openpi.policies.policy_config import create_trained_policy

        observation = {k: v for k, v in sample.items() if k != "actions"}
        if args.observation:
            with np.load(args.observation) as data:
                observation = dict(data)
            observation["prompt"] = observation["prompt"].item()
        policy = create_trained_policy(
            config.get_config(args.policy_config), args.checkpoint, sample_kwargs={"num_steps": 1}
        )
        actions = policy.infer(observation)["actions"]
        assert actions.shape == (50, 12)
        assert np.isfinite(actions).all()
        if args.save_actions:
            args.save_actions.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.save_actions, actions)
        report["inference_actions"] = list(actions.shape)
        report["smoke_denoising_steps"] = 1
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
