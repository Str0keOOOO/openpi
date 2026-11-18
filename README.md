# RoboCasa365 / OpenPI

本仓库基于 [robocasa-benchmark/openpi](https://github.com/robocasa-benchmark/openpi)，上游为 [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)。包含四任务数据准备、π0.5 训练和固定测试集评测。

## 目录

```text
workspace/
├── openpi/                      # 唯一主 Git 仓库
│   ├── src/openpi/              # 模型、训练、数据加载
│   ├── scripts/                 # 统一命令入口
│   ├── examples/                # RoboCasa 增强、回放、评测
│   ├── third_party/             # RoboCasa、RoboSuite 官方 submodule
│   └── uv.lock                  # 两套环境共用的依赖版本锁
├── data/
│   ├── train/                   # 1997 条轨迹，404039 帧
│   ├── views/{human,anchor,all}/ # 418 / 418 / 1997 条轨迹
│   └── test/                    # 20 个场景，1460 个测试状态
├── model/                       # 基础权重、训练检查点、tokenizer
├── output/                      # 运行日志、评测结果
└── envs/{policy,sim}/            # Python 环境
```

默认工作目录是本仓库的上一级。换位置后可设置 `ROBOCASA365_ROOT`；数据、模型和输出均相对此目录定位。数据和模型不提交到 Git。

官方依赖固定在 RoboCasa `425684d`、RoboSuite `5ce6643`。下载的仿真资源保留在 RoboCasa 的 assets 目录，由上游忽略规则管理。

## 安装环境

需要 `uv`，所有命令从 `openpi/` 执行：

```bash
git submodule update --init third_party/robocasa third_party/robosuite
bash scripts/setup.sh all
source scripts/env.sh
```

安装 Python 3.11 的两套环境：

- `envs/policy`：OpenPI、JAX 0.5.3、Flax 0.10.2、PyTorch 2.7.1，以及训练数据读取依赖。
- `envs/sim`：RoboCasa、RoboSuite、MuJoCo 3.3.1 和评测客户端。

两者均使用 NumPy 2.2.5 和 LeRobot 0.3.3，共用一份 `uv.lock`；仿真环境只安装 `sim` 依赖组。策略环境也安装仿真包，因为上游训练配置会导入 RoboCasa。`setup.sh policy` 或 `setup.sh sim` 可单独安装其中一套。CUDA 依赖由锁文件安装，GPU 运行仍需要主机的 NVIDIA 驱动。

已有仿真资源无需重新下载；新机器可在仿真环境执行：

```bash
../envs/sim/bin/python -m robocasa.scripts.download_kitchen_assets --type all
```

RoMa、DINOv3、SHARP 的标签生成工具已移出。双教师网络结构和已有缓存的读取代码保留，用于已有 checkpoint 的兼容；当前没有完整教师缓存，不提供重新生成教师标签的安装流程。

## 数据准备

`data/train/episode_index_map.csv` 的 `type` 区分 Human、Anchor、单项扰动和多项扰动。保留的 418 条 Human、418 条 Anchor 和 1161 条扰动轨迹合计 1997 条。

```bash
scripts/run.sh prepare
```

`human/<任务>` 和 `anchor/<任务>` 可直接用 LeRobot 0.3.3 打开。每个子集只包含选中的轨迹，轨迹和全局帧都从 0 编号。`meta/episodes.jsonl` 中的 `train_episode_index` 对应 `data/train` 中的原编号，可用来查配对表。`all/<任务>` 直接引用完整数据。

Anchor 包含 415 条生成轨迹和 3 条 Human 副本。`data/train/human_anchor_pairs.csv` 用 `anchor_generation_method=human_copy` 标出副本。它们在 Anchor 视图中各自存有独立的 Parquet、三路视频和仿真状态，状态和动作列分别为 16 维和 12 维定长数组。其余轨迹的视频和仿真状态通过相对链接复用。

三个视图的 `meta/stats.json` 沿用同任务的 Human 归一化参数。首次在新目录构建时，用 `--normalization-root` 指向现有 Human 视图。

## 训练

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run.sh train pi05_robocasa_human418_baseline --exp-name=human_run
```

按实际分配的 GPU 修改 `CUDA_VISIBLE_DEVICES`。其他当前配置：

| 配置 | 数据 | 步数 |
|---|---|---:|
| `pi05_robocasa_human418_baseline` | `views/human` | 默认 15000，可用 `--num-train-steps=6000` |
| `pi05_household_anchor_lora_6k` | `views/anchor` | 6000 |
| `pi05_household_augmented_lora_30k` | `views/all` | 30000 |

新训练从 `model/pi05_base/params` 初始化，保存到 `model/checkpoints/<配置>/<实验名>/`。保留下来的推理检查点不一定包含优化器状态，恢复训练需使用完整训练断点。

训练日志默认离线写入 `output/wandb/`，无需登录；需要同步到 W&B 时设置 `WANDB_MODE=online`。

## 策略服务与评测

一个终端启动策略：

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run.sh serve \
  --policy-config pi05_robocasa_human418_baseline \
  --checkpoint ../model/pi05_robocasa_human_baseline/pi05_human418_baseline_gpu13/6000 \
  --port 8000
```

另一个终端运行评测：

```bash
scripts/run.sh evaluate \
  --policy-config pi05_robocasa_human418_baseline \
  --checkpoint ../model/pi05_robocasa_human_baseline/pi05_human418_baseline_gpu13/6000 \
  --port 8000 --gpu 0 --limit 1
```

去掉 `--limit 1` 执行完整 1460 个测试；`--dataset`、`--output` 可覆盖默认位置。`--shard-index` 和 `--shard-count` 可分片运行。每次运行默认创建独立输出目录，保留结果 CSV、汇总和回放视频。测试集本身不写入评测结果。

## 检查与下载

```bash
# 读取真实训练数据并检查三个视图
JAX_PLATFORMS=cpu scripts/run.sh check

# 检查仿真状态还原和一步执行
scripts/run.sh check-sim

# 下载官方 Human300 75000 步推理权重，支持续传
scripts/run.sh download
```

下载目标为 `model/pi05_pretrain_human300/multitask_learning/75000`。tokenizer 单独缓存在 `model/big_vision/paligemma_tokenizer.model`。

数据格式说明位于工作目录的 `data/train/README.md`、`data/views/README.md` 和 `data/test/README.md`。历史过程文件已移入工作目录外的 `.Trash`。

2026-11-18 已在当前机器验证：42 项数据视图、梯度累积、变换和客户端测试通过；真实训练样本读取正常；Human418 的 6000 步 checkpoint 在 CPU 上完成一次推理，并将前 5 步动作送回恢复的测试场景执行，渲染正常。该推理检查只用了 1 次去噪迭代，用于确认流程可运行；未重跑完整训练或 1460 条评测。
