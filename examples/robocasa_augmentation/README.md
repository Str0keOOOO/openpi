# PandaOmron 11D augmentation

`generate.py` reads a paired Human/Anchor LeRobot dataset, clones it to a separate
output, and measures every real first frame with the matching MuJoCo model.
Only Human trajectories are used as motion sources. The paired Anchor's actual
first-frame robot configuration defines the reference; CSV poses are not used.

The central SQLite scheduler dispatches isolated MuJoCo processes across the
selected GPUs. It reserves coverage for in-flight trials and updates coverage
using actual executed first states. Original Human samples retain their type and
contribute to the same coverage counts. Angles are degrees, translations metres;
XY is in the reference Anchor's planar local basis. Height is torso joint travel.

Single caps are 12 per dimension, including Human Single samples. Original data
above a cap are never removed. New Multi targets per task are 60 four-dimensional,
60 five-to-seven-dimensional and 35 eight-to-eleven-dimensional generated samples.
Existing Human and generated samples all contribute to signed amplitude density;
Human initialization noise does not substitute for intentionally varied high-dimensional combinations.
Task, signed amplitude and combination deficits determine the next candidate.
Amplitude bins are three equal intervals of each configured physical range.
Per-reference joint limits further restrict proposals. Measured deltas decide the label.
CPU-only MuJoCo screening restores the paired Anchor robot in the original Human
scene and checks robot-to-environment contacts before allocating a render worker.
The full simulator independently runs the official predicate again. Exact near-duplicate
requests are skipped; a source/sign/amplitude combination stops after two failed rollouts.

New data types are `perturb_single` and `perturb_multi`. Both link to the original
Human and reference Anchor in episode metadata, provenance and the root index.
Per-candidate JSON/video files are transient and deleted after recording trial
outcomes or exporting successful data; original datasets are never deleted.
Global SQLite/CSV logs preserve request, actual delta and failure information.

GPU UUIDs bind CUDA selection. Each GPU admits at most four workers, with a
4 GiB per-worker memory reservation and 3 GiB of headroom. Live memory queries
credit allocations owned by this generator when calculating capacity. Resource
errors cool down that GPU for five minutes. No existing GPU process is killed.
The current launch sets the elapsed-time, free-disk, attempt and consecutive
failure stop thresholds to zero, disabling them. When resources or candidates
are temporarily unavailable, the scheduler waits and retries. The agreed coverage
targets remain active. CPU screening and rollout failures are recorded separately.

Run with the RoboCasa environment and both RoboCasa/robosuite on PYTHONPATH:

```bash
python examples/robocasa_augmentation/generate.py \
  --source /path/to/paired_dataset --output /path/to/new_dataset \
  --work /path/to/independent_logs --gpus 0 2 4 7
```

Reusing these paths resumes the SQLite ledger and reference audit. A transaction
journal protects partially exported episodes. Successful exports are reconciled
before planning additional trials. `report.py --work ...` refreshes plots using
saved CSVs without any GPU or simulation. Coverage deficits and failed attempts
remain visible if the attempt budget is exhausted.
