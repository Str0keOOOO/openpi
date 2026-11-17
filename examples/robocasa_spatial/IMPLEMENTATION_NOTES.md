# RoboCasa365 Spatial Query + SHARP

This experiment uses the four official RoboCasa Pretraining Human tasks and a
fixed-base subset of their 50-step EE action windows. Original Human datasets
and pretrained π0.5 parameters are read-only. The registered training config is
`pi05_robocasa_spatial_query_human4`.

## Data and supervision

- Four tasks, 100 selected episodes per task. Five moving episodes are removed
  by the base-motion filter; the user exclusion list additionally removes
  PickPlaceCounterToCabinet EP38, EP45, and EP103 for low-texture Anchor views.
  The active training set is 392 episodes. Source Human files remain untouched.
- All eight exclusions are explicit in
  `/data5/xuchenfei/robocasa365/outputs/anchor_visual_audit/manual_episode_exclusions.csv`:
  OpenStandMixerHead EP2/14/29/43 and PickPlaceCounterToCabinet EP38/45/79/103.
  EP79 overlaps the moving-base filter, so the list adds only three new
  exclusions to the five already filtered movers.
- Dataset adapter: `episode_dataset.py`. It aligns all three unmasked student
  RGB views and State at window start `t`, actions `t:t+50`, language, and the
  frame-level teacher by `(task, episode_index, frame_index)`.
- LeRobot action order is `[base xyz, torso, mode, EE xyz, EE rotvec, gripper]`.
  The training mask supervises stored dimensions 5–11 only. `SpatialOutputs`
  maps them to native RoboCasa EE/gripper dimensions and forces arm mode with
  zero base/torso commands for fixed-base inference.
- Training normalization is computed from the post-exclusion training-window
  distribution. It is saved at
  `/data5/xuchenfei/robocasa365/outputs/anchor_visual_audit/training_norm_stats_exclude8/`.
  Do not use the 75k mixture checkpoint's normalization statistics for this
  dataset: its stored action dimension 10 has zero standard deviation although
  this dataset varies on that dimension.
- SHARP supervision is left/right Agentview only. Each high-resolution field
  stores dimensionless `(u_anchor-u_current)/W, (v_anchor-v_current)/H`; masked
  4×4 average pooling gives 32D per camera and 64D total, left then right.
  The student retains all three original RGB cameras, State, and language.
- The static field is reconstructed once from episode frame 0. Each action
  window gets a MuJoCo geometry-ID mask for the current robot and every object
  that moved during the episode. The initial and Anchor-view pixels occupied
  by those same dynamic objects are also excluded. Pooling is recomputed per
  frame start. The historical camera-motion threshold is superseded by exact
  per-window camera-pose reprojection; see the migration note at the end of this
  file.
- Provisional frame-level teacher caches are isolated from the `formal`
  training cache. Student images are never robot-masked.

## Anchor review

The visual audit lives at
`/data5/xuchenfei/robocasa365/outputs/anchor_visual_audit/`. Its resumable
`build_audit.py` batch visits the 100 selected episodes in each task and saves
the original Human images, reconstructed Human-pose views, and task-reset
Anchor candidate views. `progress.json` records current completion;
`index.html` groups the 2×3 comparisons by task, and `review_template.csv` is
the human decision file. A `rendered_candidate` is not an accepted Anchor.

`export_episode_geometry.py` runs in the RoboCasa environment and exports full
task-reset Anchor pose, each Agentview camera's current/Anchor SE(3), and robot
segmentation masks. `generate_teacher_cache_sharp.py` runs in the independent
SHARP environment and launches that exporter as needed. It saves 64D Latent,
validity, provenance, masks, flow masks, and depth/flow diagnostics. Formal
mode requires `review_status=accepted`; provisional mode writes only under a
path containing `provisional`.

The final frame-aligned stages are resumable:

```bash
# Export the exact fixed-base, 50-action windows after exclusions
../envs/benchmark/bin/python -m examples.robocasa_spatial.build_window_manifest

# In RoboCasa Python: robot and dynamic-object segmentation for each window
../envs/robocasa/bin/python -m examples.robocasa_spatial.export_dynamic_masks \
  --output-root /data5/xuchenfei/robocasa365/outputs/anchor_visual_audit/dynamic_masks/provisional_v2

# CPU: pool the existing SHARP static fields with each window's dynamic masks
../envs/benchmark/bin/python -m examples.robocasa_spatial.build_frame_teacher_cache

# Validate exact window/camera/cache alignment
../envs/benchmark/bin/python -m examples.robocasa_spatial.validate_frame_teacher_cache
```

The window cache lives in `teacher_cache/frame_level_provisional/<task>/`; its
256×256 fields are in `teacher_cache/frame_level_provisional/static_fields/`.
These caches remain `unreviewed_provisional` until each Anchor is accepted.

LeRobot `observation.state[0:3]` is the base center-site position; episode
metadata `init_robot_base_pos` is the base-root reset position. Their z values
differ by the fixed site height, so the audit compares XY/yaw and reports the
site/root vertical difference separately.

## Preflight and commands

Run from the `openpi` root with the indicated environment:

```bash
# CPU: actual 52,215 train windows and associated sampled action frames
../envs/benchmark/bin/python -m examples.robocasa_spatial.compute_spatial_norm_stats

# CPU contract tests
../envs/benchmark/bin/python -m pytest -q examples/robocasa_spatial/test_spatial_contracts.py

# RoboCasa Python environment: export one episode's camera geometry/masks
../envs/robocasa/bin/python -m examples.robocasa_spatial.export_episode_geometry \
  --task PickPlaceSinkToCounter --episode 0 \
  --output /data5/xuchenfei/robocasa365/outputs/anchor_visual_audit/geometry_staging/PickPlaceSinkToCounter_ep000000.npz

# Separate SHARP Python environment; diagnostic only until all Anchors are reviewed
CUDA_VISIBLE_DEVICES=2 ../envs/sharp/bin/python -m examples.robocasa_spatial.generate_teacher_cache_sharp \
  --provisional --limit 1

# Real checkpoint/data forward/backward/sampler smoke check; no optimizer update
CUDA_VISIBLE_DEVICES=2 ../envs/benchmark/bin/python -m examples.robocasa_spatial.validate_model_smoke

# After Anchor review: real formal lambda calibration; provisional runs require explicit --allow-provisional
../envs/benchmark/bin/python -m examples.robocasa_spatial.calibrate_spatial_lambda \
  --output /data5/xuchenfei/robocasa365/outputs/anchor_visual_audit/lambda_calibration.json
```

The registered training config has a 95% formal teacher episode-coverage gate;
it intentionally refuses to start before enough reviewed SHARP caches exist.
Current `lambda_spatial=0.1` is provisional until the real gradient-ratio
calibration runs. After exclusions, there are 58,086 valid 50-step windows:
52,215 train and 5,871 validation, split by episode. Selected schedule: global batch 32,
6,000 steps (user-selected; 192,000 sampled windows, approximately 3.677 equivalent epochs).
The exposure-matched estimate at batch 32 is 1,313 steps. The selected schedule uses
with 100 warmup steps and cosine decay over 6,000 steps. The reference mixture
sample count remains an estimate from available task lengths. See
`outputs/anchor_visual_audit/training_schedule_estimate.json`.

## Validation state

Verified so far: five CPU geometry/action-contract tests; static compilation;
registered OpenPI config parsing; real Human adapter loading; and one actual
OpenDrawer EP0 frame-aligned mask/teacher cache with 136 windows, 122 windows
having at least one valid camera and 121 having both. The real LeRobot loader
matched frame 0 RGB/State, 50×12 Action, and 64D teacher with 60 valid cells.
A prior real π0.5 checkpoint forward/backward and Action sampling check passed
using the legacy provisional episode cache, with nonzero Action-Loss-to-Query,
Spatial-Loss-to-Query, Spatial-Loss-to-GeometryHead, and Action-Expert-LoRA
gradients; a diagnostic optimizer update and native checkpoint round-trip
passed. A fresh model check against the new frame-level cache is still pending.
No formal fine-tuning has been started.

## 2026-11-07 exact-pose reprojection update

The v3 frame cache treated any Agentview translation over 1 mm or rotation over
0.1° as an invalid camera. That branch discarded 7,645 windows. It is no longer
the active generation path. `export_window_camera_poses.py` now exports each
window's full left/right `T_world_opencv_camera` from its exact saved MuJoCo
state. `generate_reprojected_frame_teacher_cache.py` reuses the same episode's
cached SHARP depth/alpha. It back-projects static source depth points, transforms
them with each exact source-to-window camera pose, forward-splats into the current
view with a z-buffer, and projects the correspondences into the fixed Anchor view.
It then applies source/current/Anchor robot and dynamic-object masks plus alpha,
depth, visibility, occlusion, and image-bound validity before 4×4 masked average
pooling. It does not rerun SHARP. Pose magnitude is not a reject criterion. Newly
revealed pixels without source-view SHARP geometry remain invalid. The student-side
RGB, State, Action, sampler, and Loss definitions are unchanged.

Frame-0 transform direction was checked against the stored SHARP diagnostic
flow on OpenDrawer EP0, left Agentview: 50,593 common valid pixels, a maximum
normalized-flow difference of `9.35e-6`, and 7 mask disagreements out of
65,536 pixels from validity-boundary rounding. The new pose exporter was run
against the real 136 window starts for OpenDrawer EP0; its frame-0 camera pose
matched the previous MuJoCo export exactly, and all per-window translation /
rotation deltas matched the old diagnostic cache exactly. Frame 122, previously
discarded, has valid full camera matrices and approximately 1.16–1.26 mm
translation with 0.041° rotation. The full CPU reprojection rebuild completed
in a separate v4 cache directory; the old v3 cache remains untouched.

The rebuild generated 58,086/58,086 windows across 392/392 retained episodes,
with zero episode failures. Both Agentviews have at least one valid pooled
dimension for all windows; 3,487,742 of 3,717,504 teacher dimensions are valid.
All 7,645 windows that the v3 pose gate discarded now have valid dimensions from
both views. The full validator found no episode, frame-index, transform, shape,
or finite-value errors. The spatial training config now points to
`teacher_cache/frame_level_reprojected/`; student inputs, sampler, actions, losses,
and training hyperparameters were not changed. See
`outputs/anchor_visual_audit/reprojection_update_20261107.md` for per-task counts.
The cache intentionally follows the existing 58,086-row `window_manifest.csv`.
The official Groot reader also samples 19,208 episode-tail starts outside that
manifest; those still receive all-false `teacher_valid` from the existing adapter
and contribute only Action Loss, unchanged by this update.

The 400/400 Human / reconstructed-Human / Anchor comparison panels are present
in `outputs/anchor_visual_audit/index.html`. Four OpenStandMixerHead panels
(EP2/14/29/43) exposed a base-pose rendering bug: the renderer had treated the
target Anchor yaw as the compiled root frame. The helper now solves the target
pose against MuJoCo's current mobile-base kinematics and checks the realized
XY/yaw. All four panels have been regenerated and both views now show the same
kitchen and mixer. Full per-episode camera/mask/pose geometry export completed
for all 400 episodes. `geometry_pose_verification.json` reports 400/400 valid,
no mismatches, maximum realized Anchor XY error `1.76e-11 m`, and maximum yaw
error `4.20e-10 deg`. The check compares the requested reset pose with the pose
realized by MuJoCo and independently checks the exported Anchor against the
candidate table; it does not mean the candidate is accepted by a human.

The refreshed `summary.md` reports 400/400 image candidates with no scene
reconstruction failures. Eight episodes are excluded from training separately
from Anchor acceptance: four mobile-base OpenStandMixerHead demos, three
additional static low-texture cases, and overlapping mobile/low-texture
PickPlaceCounterToCabinet EP79. Their source data and review statuses are
unchanged; other Anchor candidates remain `uncertain` until reviewed.

Dual-Agentview SHARP provisional generation originally completed on GPU0 for
all 395 fixed-base episodes. After exclusions,
`teacher_cache/provisional/validation.json` checks 392 eligible caches and
reports 392/392 valid, 0 invalid. Three previously generated caches for the
newly excluded static episodes remain on disk for traceability; the adapter
and future cache-generation runs skip them. Per-task active coverage is
100/100 PickPlaceSinkToCounter, 100/100 OpenDrawer, 96/96 OpenStandMixerHead,
and 96/96 PickPlaceCounterToCabinet. The cache folder is about 373 MiB.
Included entries remain `unreviewed_provisional`, not formal training labels.
A one-sample gradient-ratio estimate from the legacy episode cache is recorded
in `outputs/anchor_visual_audit/provisional_lambda_diagnostic.json`; it is not
the final frame-level calibration. The formal config remains
`lambda_spatial=0.1` and is not calibrated.

The final-scheme dynamic-mask export is complete for all 392 active episodes
and 58,086 windows. Its per-episode outputs and task progress are under
`outputs/anchor_visual_audit/dynamic_masks/provisional_v2/`; aggregate
`progress.json` reports 392/392 episode files, zero missing files, zero read
errors, and zero failed exports. Frame-level cache construction is complete
at `outputs/anchor_visual_audit/teacher_cache/frame_level_provisional/`.
Validation reports 50,441/58,086 windows with at least one valid view
(86.84%) and 49,983 with both views (86.05%). The cache-level report is
`outputs/anchor_visual_audit/frame_level_teacher_report.md`, with detailed
per-episode validation CSV/JSON alongside the cache.

The execution sandbox denies access to the tmux socket, so these long-running
preprocessing jobs used managed command sessions rather than tmux. The
reproducible four-task data-adapter check passed at EP0/frame0 for each task:
three 256×256 RGB views, 16D State, 50×12 Action, and frame-aligned 64D
provisional teacher were joined correctly. Its result is
`outputs/anchor_visual_audit/frame_dataset_adapter_validation.json` and it can
be rerun with `python -m examples.robocasa_spatial.validate_frame_dataset_adapter`.

The current runtime cannot access the NVIDIA driver. A fresh real-checkpoint
forward/gradient/sampling smoke was attempted on CPU but was interrupted after
more than 15 minutes without producing stage output; it is **not verified**.
The prior real-checkpoint GPU validation only covered the earlier episode-level
provisional cache. Human Anchor acceptance, formal cache generation,
frame-level real-model optimizer/checkpoint round-trip, formal lambda
calibration, and full formal preflight remain pending. Do not mark this
experiment READY_TO_TRAIN until those checks pass. Formal π0.5 fine-tuning has
not been started. `/data5/xuchenfei/data_household/behavior_375_five_tasks`
remains untouched. Cache-level validation is reproducible with
`python -m examples.robocasa_spatial.validate_frame_teacher_cache`.
