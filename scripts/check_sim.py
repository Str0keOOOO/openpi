"""Restore a saved test scene and execute a short simulation smoke test."""

import argparse
import json
from pathlib import Path
import sys

from openpi.shared.paths import repository_root
from openpi.shared.paths import workspace_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=workspace_path("data/test"))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--actions", type=Path, help="NPY array of policy actions from check_workspace.py")
    parser.add_argument("--save-observation", type=Path, help="Save rendered RGB/state for policy smoke testing")
    args = parser.parse_args()
    import numpy as np

    sys.path.insert(0, str(repository_root() / "examples/robocasa_unified"))
    from env import make_env
    from grouped import STATE_KEYS
    from grouped import restore
    from grouped_evaluation import load_case
    from grouped_evaluation import native_action
    from grouped_evaluation import policy_input

    entry = json.loads((args.dataset / "manifest.json").read_text())["cases"][0]
    setting, anchor, state_file = load_case(args.dataset, entry)
    environment = make_env(
        entry["case"]["task"],
        anchor["scene"],
        anchor["scene"]["seed"],
        render=args.render or bool(args.save_observation),
    )
    try:
        with np.load(state_file) as source:
            restore(environment, anchor, dict(source), reload_model=True)
        obs = environment._get_observations(force_update=True)  # noqa: SLF001 - refresh after restoring simulator state
        state = np.concatenate([obs[key] for key in STATE_KEYS])
        np.testing.assert_allclose(state, setting["observation_state"], atol=1e-7, rtol=0)
        if args.render:
            image = environment.sim.render(width=224, height=224, camera_name="robot0_agentview_left")
            assert image.shape == (224, 224, 3)
        if args.save_observation:
            args.save_observation.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.save_observation, **policy_input(environment, anchor["instruction"], 224))
        actions = np.load(args.actions)[:5] if args.actions else [None]
        for action in actions:
            control = native_action(environment, action) if action is not None else np.zeros(environment.action_dim)
            environment.step(control)
        assert np.isfinite(environment.sim.data.qpos).all()
        print(
            json.dumps(
                {"task": entry["case"]["task"], "restored": True, "steps": len(actions), "rendered": args.render}
            )
        )
    finally:
        environment.close()


if __name__ == "__main__":
    main()
