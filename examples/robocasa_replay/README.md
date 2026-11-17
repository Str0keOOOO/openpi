# Human action replay

`replay_human.py` is the single canonical replay implementation. It restores
the episode XML and saved MuJoCo state through RoboCasa's `reset_to()`, then
executes the saved actions continuously with `env.step()`. It does not reset
or refresh robot controllers after `reset_to()`. Full-episode runs and
diagnostic action segments use this same path.

The original Human dataset is read only. Each run saves the source episode,
action interval, software versions, restore checks, per-step task-success
values, simulator states, initial comparisons and an H.264 three-camera video.
In the video, the original recording is on top and the simulation replay is
below. `success_ever` and `success_final` refer to continuous action execution;
`recorded_last_state_success` is a separate saved-state diagnostic.

Example full replay:

```bash
python replay_human.py --dataset /data5/xuchenfei/robocasa365_household \
  --output /data5/xuchenfei/robocasa365/outputs/replay_run \
  --task PickPlaceCounterToCabinet --episode 33 --gpu 1
```

Example segment replay, using source actions 80 through 139:

```bash
python replay_human.py --dataset /data5/xuchenfei/robocasa365_household \
  --output /data5/xuchenfei/robocasa365/outputs/replay_segment \
  --task PickPlaceCounterToCabinet --episode 33 --gpu 1 \
  --start-index 80 --stop-index 140
```

`diagnostic_replay.py` is a compatibility entrypoint that forwards to
`replay_human.py`; it contains no separate replay behavior. Retry and batch
drivers also invoke the canonical script. Archived run directories retain
their copied scripts as records of the implementation used at that time.
