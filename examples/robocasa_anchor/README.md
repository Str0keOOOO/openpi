# Anchor 数据处理

这里保留 Anchor 姿态、过渡轨迹、操作段回放和 LeRobot 导出实现。主要入口是 `run_batch.py`、`preflight_continuous.py`、`export_lerobot.py` 和 `retime_batch.py`，各入口通过 `--help` 查看参数。

在仓库根目录执行：

```bash
source scripts/env.sh
../envs/sim/bin/python examples/robocasa_anchor/run_batch.py --help
```

默认源数据为工作目录 `data/train`；生成结果用命令行指定输出目录。当前训练直接读取整理好的数据，流程见主 README。
