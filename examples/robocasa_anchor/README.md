# RoboCasa Human Anchor generation

The batch runner copies the cleaned Human dataset to a new output root and appends only successful, executed Anchor rollouts. Original Human episodes are not modified.

## Validity checks

At the exact CSV Anchor pose, after `sim.forward()`, `preflight.py` calls RoboCasa's `robocasa.utils.env_utils.detect_robot_collision()`. This is the same start-pose predicate used by RoboCasa's randomized base placement: any current MuJoCo contact between a robot geometry and a non-robot geometry rejects that exact candidate. The CSV pose is never resampled or shifted. Contact pairs are saved in `result.json`.

The official predicate is applied only to the initial Anchor configuration. It is not applied throughout manipulation because intentional gripper/object contacts would also trigger it. During transition and operation, deep MuJoCo contacts and non-robot scene motion are retained as diagnostic events; these diagnostics do not independently reject an otherwise successful rollout. Transition tracking and the task's success predicate remain required.

## Batch

`run_batch.py` is resumable by per-episode `result.json`. `--retry-operation-failures` overwrites the existing failed simulation outputs and retries those episodes; successful dataset entries are deduplicated by source task and episode. Original Human data remains read-only.

## EE tracking

The existing generator uses measured Human EE world poses from FK of the recorded MuJoCo states. Each source OSC goal supplies feedforward for one control step. The resulting actual pose is compared with recorded `state[t+1]`; a limited number of feedback steps correct position or rotation error before advancing. Source gripper/torso commands execute once, and feedback steps send zero incremental commands to hold their targets.

Defaults: 5 mm position tolerance, 0.04 rad orientation tolerance, at most three feedback steps per source frame, and at most 30% additional operation steps. Correction time is distributed over source progress so early errors cannot consume the entire budget. Unreachable or contact-sensitive executions may still fail. No simulated object poses are edited to force success.

`operation_actual.npz` stores aligned pre-action observations, actual actions, source-frame indices, measured next-pose references, and post-action tracking errors. `operation_result.json` records median/p95/maximum error and the executed-to-source time ratio. The absent final next-state reference is explicitly masked; no missing reference is fabricated.
