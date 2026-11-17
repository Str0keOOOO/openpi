# openpi
This fork extends [robocasa-benchmark/openpi](https://github.com/robocasa-benchmark/openpi), based on the original [openpi code](https://github.com/Physical-Intelligence/openpi) from the Physical Intelligence team.

The local extensions include household data augmentation and anchor generation, grouped RoboCasa evaluation, and spatial-query / dual-teacher training. Their workflows are in `examples/robocasa_anchor`, `examples/robocasa_augmentation`, `examples/robocasa_unified`, `examples/robocasa_spatial`, and `examples/robocasa_dual_teacher`.

Custom training configurations resolve data, weights, and outputs from `ROBOCASA365_ROOT`, defaulting to the workspace containing `third_party/openpi`. The workspace uses `data/`, `model/`, and `output/` directories. Some experiment scripts retain paths from the original workspace and need adjustment on another machine. Datasets, checkpoints, simulation assets, and external teacher models are supplied separately.

## Recommended system specs
For training we recommend a GPU with at least 80 Gb of memory (H100, H200, etc).
For inference we recommend a GPU with at least 8 Gb of memory.


## Installation
```
git clone https://github.com/Str0keOOOO/openpi
cd openpi
pip install -e .
pip install -e packages/openpi-client/
```

## Key files
- Training: [scripts/train.py](scripts/train.py)
- Evaluation: [scripts/serve_policy.py](scripts/serve_policy.py) and [examples/robocasa/main.py](examples/robocasa/main.py)
- Setting up configs: [src/openpi/training/config.py](src/openpi/training/config.py)

## Experiment workflow
```
# train model
XLA_PYTHON_CLIENT_MEM_FRACTION=1.0 python scripts/train.py \
<dataset-soup> \
--exp-name=<exp-name>

# evaluate model
# part a: start inference server
python scripts/serve_policy.py \
--port=8000 policy:checkpoint \
--policy.config=<dataset-soup> \
--policy.dir=<checkpoint-path>

# part b: run evals on server
python examples/robocasa/main.py \
--args.port 8000 \
--args.task_set <task-set> \
--args.split <split> \
--args.log_dir <checkpoint-path>

# report evaluation results
python examples/robocasa/get_eval_stats.py \
--dir <checkpoint-path>
```
