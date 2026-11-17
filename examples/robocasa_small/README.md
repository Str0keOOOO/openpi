# 固定场景的四任务 π0.5 小测评

这是一套新实验，不是官方 Atomic-Seen 成绩的替代。只完成代码，尚未运行仿真、模型推理或物理有效性验证。

## 固定内容

- 任务：`PickPlaceSinkToCounter`、`OpenDrawer`、`OpenStandMixerHead`、`PickPlaceCounterToCabinet`。
- 暂定厨房 layout=11、style=11，物品使用 pretrain split，环境初始化 seed=7。
- 当前没有下载官方 Human300 的逐演示场景元数据。因此 11/11 是明确标记的暂定值，**没有声称它是官方最常用场景，也没有保证固定它会提高成功率**。
- 每个任务在同一个厨房各生成一个原点，任务本身要求的初始物品和设备配置可以不同。每个任务内部的所有扰动使用同一模型 XML、完整仿真状态、物品、指令和相机安装参数。
- 场景不适配某任务时报告错误，不偷偷换场景、物品或 seed。实际使用前需先确认四个原点的画面与物理有效性。
- 使用官方 `pi05_pretrain_human300/multitask_learning/75000`。三路图像、224 输入尺寸、每次执行前 5 步、12 维动作、官方 horizon 与成功条件沿用原测评逻辑。
- `split=None` 并显式指定 layout/style 和 `obj_instance_split=pretrain`：上游 `split=pretrain` 会覆盖指定的场景参数，因此不能直接同时传 pretrain 和固定场景。

## 扰动定义

`benchmark.json` 默认 `sampling.mode=stratified_random`。每任务包含 1 原点、24 个身体分组（4轴×2方向×3档）、42 个机械臂分组（7关节×2方向×3档）。每个分组最多尝试50个候选，收集到5个不同的有效配置即停止；每个配置只跑1次操作。原点单独重复5次。因此四任务最多1320个扰动配置加4个原点，共1324个不同配置槽位、最多1340次操作。未采满的分组记录实际数量，不跨档补齐、不把缺失当操作失败。

| 参数（正负方向分别采样） | Small绝对幅度 | Medium绝对幅度 | Large绝对幅度 |
|---|---|---|---|
| 底盘 x、y | 0～5 cm | 5～10 cm | 10～20 cm |
| 底盘 yaw | 0～5° | 5～10° | 10～20° |
| 躯干高度 | 0～2 cm | 2～4 cm | 4～6 cm |
| 单个臂关节 | 0～2° | 2～5° | 5～10° |

每个分组在指定绝对幅度区间均匀采样，符号固定。每次只改变一个轴/关节：x为原点底盘的局部+X，y为局部+Y，yaw绕+Z正向，h为竖直上移。PandaOmron只有一条7关节机械臂，逐关节独立采样，不同时随机7个关节，不修改夹爪。身体组臂关节角保持不变；臂组底盘和躯干保持不变，并断言其他物体qpos未变。

每次恢复同一原点后应用候选，检查关节限位和官方起点碰撞。拒绝时只在同方向、同档重抽；同一个分组的全部5个有效配置共享总共50次尝试上限，绝不是每个配置各50次。通过检查立即保存该配置，累计5个就停止。尝试到50次时，已找到的配置仍保留，其余槽位记sampling_exhausted。绝不缩小幅度、换场景或用小档补大档。

每组使用由采样seed、任务、自由度/方向/档位生成的独立随机流，每个attempt也有确定的子流。所有候选写入组级attempts.jsonl，断点继续时先恢复已经通过的候选和已用次数，不重置50次预算。每个通过的配置保存具体值、完整qpos和三路图片；评测时读取保存配置，不重采，并核对恢复qpos。

`sampling.mode=fixed_sweep` 可改用固定数值扫描，此时读取 `body_perturbations` 和 `arm_perturbations.delta_deg`。同样先执行 sample 检查并保存明确值，但不重抽。修改参数应使用新输出目录。

PandaOmron 的躯干竖直移动关节为 `joint_torso_height`，范围 `[0,0.34] m`。若原点在升降下限，负向高度可能全部越界，保留原请求和明确原因，不裁剪数值。

相机安装的局部外参不变。相机安装于机器人/腕部，改变底盘、躯干或关节后，相机的**世界位姿和画面自然会变化**；不要求它的世界位姿固定。

## 起点与测评

1. `prepare` 仅为四个任务生成并保存原点。保存 XML、MuJoCo 完整 state、ctrl、episode 元数据、相机模型参数、关节映射、环境随机状态和三路开局图像。
2. `sample` 只启动仿真，不加载π0.5。逐槽位采样和筛选，保存配置、首帧、所有尝试及拒绝原因。之后 `evaluate` 每个任务只创建一次环境，校对 Python 任务对象与原点元数据，加载对应 XML；每个候选恢复同一份仿真状态和控制器，再施加一次扰动。恢复后没有额外步进、重力稳定或预操作，首帧通过 `force_update` 获取。
3. 有效性检查包括关节限位，并直接调用官方 `robocasa.utils.env_utils.detect_robot_collision()` 判定起点碰撞。先 `sim.forward()` 更新接触，再按官方 robot geometry 集合与外部 geometry 的接触返回值判断，不额外增加深度阈值，也不减掉原点接触。接触明细仍保留供诊断；自接触不被这项官方函数单独判为失败。这是官方起点接触筛选，**不是完整自碰撞、操作可达性或动力学稳定性证明**。原点和扰动条件都使用同一检查，不把无效起点标为策略操作失败；不重新随机位置、不缩小指定扰动。保存 `official_collision` 和调用函数路径，CSV 同时列出检查结果。每次加载 XML 后重建官方 geometry ID 缓存，避免使用上次模型的 ID。
4. 进入操作后按官方成功条件评测；没有新增碰撞提前终止。完整 chunk、实际执行动作与策略 state 保留。H.264 视频沿用官方方式：每两步取一帧，按20fps保存；在20Hz控制频率下，视频播放约为仿真二倍速。视频还额外保留操作前首帧。
5. 每个不同扰动配置默认1次（`repeats=1`），每组收集5个不同配置；原点单独重复5次（`baseline_repeats=5`）。相同原点多次测评不会重新随机场景。服务端有自己的采样 RNG，目前 websocket 没有每次请求重置采样种子的接口，不能把环境 seed=7 解释成每条独立推理 seed=7。未声明相同重复一定动作相同。
6. 支持 `--task`、`--case baseline`。已有 result.json 跳过；`--retry-errors` 重试报错项。不覆盖原点，配置变更必须用新输出目录。断点重启不会恢复服务端采样 RNG 流，因此不承诺与不中断的服务逐位相同。

## 后续启动命令（本轮未执行）

使用空闲 GPU 和独立端口；不要连接已有 Atomic-Seen 服务。以下 GPU 仅为使用者选择的示例，启动前检查实际占用。

```bash
cd /data5/xuchenfei/robocasa365/openpi
export GPU=4
export PORT=18281
export OUTPUT=/data6/xuchenfei/output_household/robocasa365/small_fixed_v1
bash examples/robocasa_small/run.sh plan
bash examples/robocasa_small/run.sh prepare
bash examples/robocasa_small/run.sh sample
tmux new-session -d -s robocasa_small_policy \
  "GPU=$GPU PORT=$PORT OUTPUT=$OUTPUT bash /data5/xuchenfei/robocasa365/openpi/examples/robocasa_small/run.sh serve"
# 服务日志确认模型就绪后再启动；不停止任何其他服务。
tmux new-session -d -s robocasa_small_eval \
  "GPU=$GPU PORT=$PORT OUTPUT=$OUTPUT bash /data5/xuchenfei/robocasa365/openpi/examples/robocasa_small/run.sh evaluate"
```

## 根据官方数据选择常见厨房

下载官方数据后可先运行 CPU 工具：

```bash
python examples/robocasa_small/select_scene.py /path/to/official/datasets \
  --output /tmp/robocasa_scene_frequency.json
```

工具统计提供目录中 `ep_meta.json` 的布局/风格频次，提出最高频组合，频次相同时按编号排序。不会自动修改配置。若没有元数据则明确退出。频次代表所提供数据，若只有一个子集不能宣称是整个 Human300 的分布。实际场景还需要包含四个任务所需设施；不根据测评结果筛选“高分 seed”。

## 输出

输出在使用者指定的 data6 目录，源代码目录不存实验文件。`anchors/<Task>/` 保留初态；`configurations/<Task>/<slot>/` 保存通过的具体配置、state.npz和三路首帧；`sampling_attempts/<Task>/<group>/attempts.jsonl` 保存组内全部候选及拒绝原因，`sampling_groups.csv` 汇总各组实际尝试次数和有效配置数量；未找到的槽位明确记录 sampling_exhausted。`sampling_summary.csv` 包含分档、方向、实际幅度、attempt数与拒绝计数，应与操作成功率一起查看，防止条件筛选造成的分布偏差。`trials/<Task>/<case>_rN/` 保留各条件的 requested delta、实际 qpos/位姿、contacts、有效性、三路初始图、轨迹与视频；`results.csv`、`summary.json` 汇总已完成/无效/错误计数。`service.json` 中 checkpoint 是声明路径，客户端不能单凭 websocket 元数据校验远端权重文件，因此必须使用本目录 `serve` 启动指定检查点。

后续模型复用同一配置池：用新 output 保存结果，在 evaluate 传 `--starts-root /path/to/original/output`。可更换 checkpoint/推理参数/重复次数，原点与采样相关配置必须一致，并校对 start_hash；读取原配置池不修改它。
