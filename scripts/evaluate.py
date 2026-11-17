"""Evaluate a running policy on the fixed RoboCasa test set."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

from openpi.shared.paths import repository_root
from openpi.shared.paths import workspace_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=workspace_path("data/test"))
    parser.add_argument(
        "--output",
        type=Path,
        default=workspace_path("output", "eval_" + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy-config", required=True)
    parser.add_argument("--model", default="policy")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--limit", type=int, help="Run only the first N cases")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("Require 0 <= shard-index < shard-count")
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(repository_root() / "examples/robocasa_unified"))
    from grouped_evaluation import evaluate
    from grouped_evaluation import report

    config = json.loads((args.dataset / "config.json").read_text())
    config["models"] = {args.model: {"checkpoint": str(args.checkpoint.resolve()), "policy_config": args.policy_config}}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    evaluate(
        config,
        args.output,
        args.model,
        args.port,
        dataset=args.dataset,
        gpu=args.gpu,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        max_episodes=args.limit,
    )
    report(config, args.output)
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
