# 当前训练输入

使用官方 GrootOpenpiMultiDataset。它负责 N^0.4 任务采样、Episode/帧采样、
三路 RGB、State、Action、语言读取，以及末尾 Action Chunk 重复末帧补齐。
不存在额外底盘运动/控制模式窗口筛选，不再丢弃末尾不足 50 步的起点。

392 条演示全部用于训练，共 77,294 个帧起点，无验证留出集。
四任务 Episode 数为 100、100、96、96。groot/ 下只有选定 Episode 的原生元数据
和源数据文件链接，避免再次执行 100_demos 随机过滤后补入其他演示。

教师附件只按 task/episode/frame 查询 64D 缓存，不负责采样或读取动作。
末尾 49 个起点现已按各自真实帧生成动态 Mask 和教师特征；训练配置读取
`teacher_cache/frame_level_all_starts`，覆盖全部 77,294 个起点。教师无效维度仍由
`teacher_valid` 屏蔽，不会借用其他帧标签。旧 window_manifest.csv 不控制训练采样。

State/Action 统计由官方 DataConfig 加载，采用官方 EE 在前的维度顺序。
当前 batch=32、steps=6000、lambda_spatial=0.1、EMA=0.99、peak_lr=2.5e-5。
未启动正式训练。旧自定义窗口加载代码及相关工具归档于 legacy_custom_loader/。

正式训练配置：`pi05_robocasa_spatial_query_human4`。
当前入口：`python scripts/train.py pi05_robocasa_spatial_query_human4 --exp-name=pi05_human4_spatial_sharp`（未执行）。

## 精确相机位姿视差缓存更新

旧 `robocasa-frame-teacher-v3` 使用 episode 初始像素视差，并在相机变化
超过 1 mm / 0.1° 时屏蔽窗口。新版本取消该门槛：每个窗口使用记录的左右
Agentview 精确位姿，将 episode 初始 SHARP 深度反投影为三维点，再分别投影到
当前窗口相机和固定 Anchor 相机；三维点的相机变换产生当前到 Anchor 的归一化
像素位移。当前视角使用双线性前向投影和 z-buffer。保留源/当前/Anchor 的机器人、
动态物体、可见性、遮挡及无效投影 Mask，再做 4×4 Masked Average Pooling。
位姿幅度不会使整窗无效；初始视角没有重建到的新增显露区域仍标为无效。

新增流程：

```bash
# CPU：从记录的底盘完整位姿恢复相机矩阵，并与全状态 MuJoCo 相机位移统计交叉核对
../envs/benchmark/bin/python -m examples.robocasa_spatial.export_window_camera_poses

# CPU：复用已有 SHARP 深度/Alpha，按窗口真实位姿生成精确对齐的 64D 教师
../envs/benchmark/bin/python -m examples.robocasa_spatial.generate_reprojected_frame_teacher_cache

# CPU：检查相机矩阵、帧索引、Latent 和有效性 Mask
../envs/benchmark/bin/python -m examples.robocasa_spatial.validate_reprojected_frame_teacher_cache
```

生成器写入独立的 `frame_level_reprojected/` v4 目录，不覆盖旧缓存；末尾帧补充结果
合并到 `frame_level_all_starts/`。四任务共 392 条演示、77,294 个起点均有教师记录，
进度和失败记录保留在对应输出目录。旧
`build_frame_teacher_cache.py` 已停用，防止重新生成基于固定像素视差的 v3 标签。
