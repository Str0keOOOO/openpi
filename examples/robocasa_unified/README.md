# Three-model shared PandaOmron benchmark

All modes require an explicit independent output directory. `plan` imports no
simulator, model or GPU code. No launch script automatically starts evaluation.

The 820 slots/model comprise 40 baseline, 660 Single and 120 Multi tests.
Single signs are interleaved 3/2 or 2/3 within five samples, never five per sign.
Height is positive only. Small/medium/large magnitudes are 0–0.5, 0.5–1.0,
1.0–1.5 times the augmentation bounds: 5/10/15 cm or 5/10/15 degrees.
The 3 Multi groups change exactly 2, 3–4, or 5–11 effective dimensions.
Normal has normalized infinity norm at most 1; strong is greater than 1 and at
most 1.5. Every active coordinate exceeds the declared physical tolerance.
No separate OOD test is included.

## Actual references and scene isolation

Set `anchors_manifest` in a copy of benchmark.json using the example schema.
Each entry identifies an actual saved Anchor first frame, never CSV coordinates.
The sampler restores its model.xml.gz, ep_meta.json and states.npz[0].
It rejects training/test layout-style overlap; this is a deliberately explicit
scene-isolation rule, not a claim that changing an episode seed alone is enough.
The existing training-derived 418 Anchors are not silently reused as held-out
tests. No held-out Anchor set has yet been selected in the default config.

`sample` checks actual magnitudes, joint limits and official initial collision;
it never runs the policy or selects on task success. Up to 50 attempts are
allowed per slot, preserving its sign and magnitude stratum. A slot without a
legal sample is reported invalid; it is not counted as policy failure and is
not shrunk into another band. Every accepted state is saved once for all models.
Controller initialization must leave physical qpos unchanged. No hidden physics
settling or IK compensation changes the requested robot configuration.

## Commands (not run automatically)

```bash
python examples/robocasa_unified/main.py --mode plan --output /new/fixed_testset
python examples/robocasa_unified/main.py --config /edited/benchmark.json --mode sample --output /new/fixed_testset
# Optional parallelism: --shard-index 0 --shard-count 5 (one index per worker).
python examples/robocasa_unified/seeded_server.py --policy-config CONFIG --checkpoint CHECKPOINT --port 18401
python examples/robocasa_unified/main.py --config /edited/benchmark.json --mode evaluate --output /new/fixed_testset --model M1 --port 18401
python examples/robocasa_unified/main.py --config /edited/benchmark.json --mode report --output /new/fixed_testset
```

Set frozen checkpoint/config entries for M1/M2/M3 before evaluation; M2/M3
training data must be identical. Inference is independent, never connected to an
existing unrelated policy service. The seeded server resets the JAX sampling key
per chunk, so processing order and resume do not change that chunk's random key.
All models use the same three input cameras, fixed starts, seeds, task horizon,
five-action replanning interval and native info['success'] criterion.

Completed result.json files are skipped on resume; error files can be rerun.
Missing/invalid configurations remain separate from completed policy trials.
Reports retain success/count/percentage tables plus task, family, individual
joint and level CSVs. The plan count is not a certificate of 820 physically
validated starts; consult sampling_summary.json after actual sampling.
