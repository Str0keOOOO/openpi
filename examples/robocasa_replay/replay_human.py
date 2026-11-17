"""Independent, resumable Human action playback; source datasets are read only.

The supervisor owns episode subprocesses, never existing services. Each episode
loads its saved XML and initial MuJoCo state once, then executes official actions
without state corrections. Loading the saved final state is a separate diagnostic.
"""
import argparse
import concurrent.futures
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

TASKS = ["PickPlaceSinkToCounter", "OpenDrawer", "OpenStandMixerHead",
         "PickPlaceCounterToCabinet"]
CAMERAS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def replay(args):
    import gzip
    import xml.etree.ElementTree as ET
    import cv2
    cv2.setNumThreads(1)
    import numpy as np
    import pandas as pd
    import imageio.v2 as imageio
    import robosuite
    import robocasa
    import mujoco
    from robocasa.utils import lerobot_utils as LU
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

    root = args.dataset / args.task / "lerobot"
    dest = args.output / "episodes" / args.task / f"episode_{args.episode:06d}"
    dest.mkdir(parents=True, exist_ok=True)
    row = dict(task=args.task, episode_index=args.episode, status="running",
               source_dataset=str(root), phase="read_data", gpu=args.gpu)
    started = time.perf_counter()
    timings = {}
    env = writer = None
    captures = []
    try:
        source_id = next(int(r["source_episode_index"]) for r in csv.DictReader(
            (args.dataset / "episode_index_map.csv").open())
            if r["task"] == args.task and int(r["filtered_episode_index"]) == args.episode)
        row["source_episode_index"] = source_id
        meta = LU.get_episode_meta(root, args.episode)
        xml = LU.get_episode_model_xml(root, args.episode)
        states = LU.get_episode_states(root, args.episode)
        actions = LU.get_episode_actions(root, args.episode)
        parquet = next((root / "data").glob(f"*/episode_{args.episode:06d}.parquet"))
        table = pd.read_parquet(parquet)
        recorded_obs = np.stack(table["observation.state"].to_numpy())
        if not (len(states) == len(actions) == len(recorded_obs)):
            raise ValueError("state/action/observation lengths do not match")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError("Nonfinite saved states or actions")
        full_action_count = len(actions)
        action_start = args.start_index
        action_stop = full_action_count if args.stop_index is None else args.stop_index
        if not 0 <= action_start < action_stop <= full_action_count:
            raise ValueError(
                f"Invalid action segment [{action_start}, {action_stop}) for {full_action_count} actions"
            )
        states = states[action_start : action_stop + 1]
        actions = actions[action_start:action_stop]
        recorded_obs = recorded_obs[action_start : action_stop + 1]
        info = json.loads((root / "meta/info.json").read_text())
        envmeta = LU.get_env_metadata(root)
        row.update(steps_planned=len(actions), source_action_start=action_start,
                   source_action_stop=action_stop, controller_initialization="official_reset_to_only",
                   layout_id=meta["layout_id"],
                   style_id=meta["style_id"], instruction=meta["lang"],
                   fixture_refs=meta.get("fixture_refs", {}),
                   xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
                   recording_versions={k:envmeta.get(k) for k in
                     ("env_version", "robosuite_version", "mujoco_version")},
                   runtime_versions=dict(robocasa=robocasa.__version__,
                     robosuite=robosuite.__version__, mujoco=mujoco.__version__))
        kw = dict(envmeta["env_kwargs"])
        kw.update(env_name=envmeta["env_name"], has_renderer=False,
                  has_offscreen_renderer=True, use_camera_obs=False,
                  camera_names=CAMERAS, camera_widths=256, camera_heights=256,
                  render_gpu_device_id=args.gpu, seed=0, ignore_done=True,
                  horizon=max(len(actions) + 10, 1000))
        row["phase"] = "restore"
        dump(dest / "progress.json", row)
        t0 = time.perf_counter()
        env = robosuite.make(**kw)
        reset_to(env, dict(model=xml, states=states[0], ep_meta=json.dumps(meta)))
        # Fixture geometric predicates read Python-side fixture transforms. Keep
        # these aligned with the episode XML, without editing the saved model.
        bodies = {b.get("name"): b for b in ET.fromstring(xml).findall("./worldbody/body")}
        for name, fixture in env.fixtures.items():
            body = bodies.get(name + "_main")
            if body is not None and hasattr(fixture, "_obj"):
                for key in ("pos", "quat", "euler"):
                    if key in body.attrib:
                        fixture._obj.set(key, body.get(key))
                    else:
                        fixture._obj.attrib.pop(key, None)
        # Follow official action playback: reset_to() restores the episode;
        # execute recorded actions without an extra controller synchronization/reset.
        env.timestep, env.done = 0, False
        env.cur_time = float(env.sim.data.time)
        env._obs_cache = {}
        env._reset_observables()
        env.sim.forward()
        actual0 = env.sim.get_state().flatten().copy()
        if actual0.shape != states[0].shape:
            raise ValueError("Restored simulator dimension differs from recording")
        state_error = float(np.max(np.abs(actual0 - states[0])))
        if state_error > 1e-9:
            raise ValueError(f"Initial simulator state restore error: {state_error}")
        for key, value in meta.get("fixture_refs", {}).items():
            actual = env.fixture_refs.get(key)
            if isinstance(actual, tuple):
                actual = actual[0]
            if actual is None or actual.name != value:
                raise ValueError(f"Target fixture mismatch: {key}: {actual} != {value}")
        row.update(initial_restore_ok=True, initial_sim_state_max_abs_error=state_error,
                   control_freq=float(env.control_freq), physics_timestep=float(env.sim.model.opt.timestep),
                   video_fps=float(env.control_freq) / args.video_skip,
                   initial_success=bool(env._check_success()))
        if abs(float(env.control_freq) - float(info["fps"])) > 1e-6:
            raise ValueError("Dataset FPS differs from environment control frequency; resolve before playback")
        timings["restore_seconds"] = time.perf_counter() - t0

        def observation():
            ob = env._get_observations(force_update=True)
            return np.concatenate([ob[k] for k in
                ("robot0_base_pos", "robot0_base_quat", "robot0_base_to_eef_pos",
                 "robot0_base_to_eef_quat", "robot0_gripper_qpos")])

        def render():
            return [np.ascontiguousarray(env.sim.render(width=256, height=256,
                    camera_name=c)[::-1]) for c in CAMERAS]

        for c in CAMERAS:
            path = root / "videos" / f"chunk-{args.episode // 1000:03d}" / f"observation.images.{c}" / f"episode_{args.episode:06d}.mp4"
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open original video {path}")
            captures.append(cap)
            for _ in range(action_start):
                if not cap.grab():
                    raise RuntimeError(f"Cannot seek original video to action {action_start}: {path}")

        def originals():
            frames = []
            for cap in captures:
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError("Original video ended before recorded trajectory")
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            return frames

        def panel(top, bottom, label):
            canvas = np.full((560, 768, 3), 248, dtype=np.uint8)
            canvas[24:280] = np.concatenate(top, axis=1)
            canvas[304:560] = np.concatenate(bottom, axis=1)
            cv2.putText(canvas, f"DATASET | {args.task} EP{source_id} | {label}",
                        (7, 17), cv2.FONT_HERSHEY_SIMPLEX, .43, (25, 25, 25), 1)
            cv2.putText(canvas, "SIM ACTION REPLAY | left agentview / right agentview / wrist",
                        (7, 298), cv2.FONT_HERSHEY_SIMPLEX, .43, (25, 25, 25), 1)
            return canvas

        initial_obs = observation()
        initial_images = render()
        initial_original = originals()
        imageio.imwrite(dest / "initial_comparison.png",
                        panel(initial_original, initial_images, "initial, before any action"))
        for i, cam in enumerate(CAMERAS):
            imageio.imwrite(dest / f"initial_{cam}.png", initial_images[i])
        row["initial_rgb_mae"] = {cam:float(np.abs(initial_images[i].astype(float)
                                            - initial_original[i].astype(float)).mean())
                                  for i, cam in enumerate(CAMERAS)}
        row["initial_observation_max_abs_raw_error"] = float(np.max(np.abs(initial_obs-recorded_obs[0])))
        # Quaternion comparison accounts for q and -q representing the same rotation.
        for name, sl in (("base", slice(3,7)), ("eef", slice(10,14))):
            a, b = initial_obs[sl], recorded_obs[0,sl]
            cosine = np.clip(abs(float(a@b)) / (np.linalg.norm(a)*np.linalg.norm(b)), 0, 1)
            row[f"initial_{name}_rotation_error_deg"] = float(np.degrees(2*np.arccos(cosine)))
        np.savez_compressed(dest / "initial_state.npz", sim_state=actual0,
                            observation_state=initial_obs, recorded_observation_state=recorded_obs[0],
                            qpos=env.sim.data.qpos.copy(), qvel=env.sim.data.qvel.copy())
        dump(dest / "progress.json", dict(row, phase="action_replay"))
        writer = imageio.get_writer(str(dest / "review.mp4"), format="FFMPEG",
                    fps=float(env.control_freq)/args.video_skip, codec="libx264",
                    pixelformat="yuv420p", macro_block_size=2, quality=7,
                    ffmpeg_params=["-threads", "1", "-preset", "veryfast"])
        sim_states, observations, successes, details = [], [], [], []
        errors = []
        timings.update(step_seconds=0.0, render_seconds=0.0, video_seconds=0.0)
        first_success = None
        for t, action in enumerate(actions):
            # Both rows correspond to pre-action frame t; no expert state is
            # written into the simulation after its initial restoration.
            frame = initial_original if t == 0 else originals()
            observations.append(observation())
            if t % args.video_skip == 0:
                rt = time.perf_counter()
                sim_images = initial_images if t == 0 else render()
                timings["render_seconds"] += time.perf_counter() - rt
                vt = time.perf_counter()
                writer.append_data(panel(frame, sim_images, f"pre-action frame {t}"))
                timings["video_seconds"] += time.perf_counter() - vt
            st = time.perf_counter()
            env.step(action)
            timings["step_seconds"] += time.perf_counter() - st
            simstate = env.sim.get_state().flatten().copy()
            if not np.isfinite(simstate).all():
                raise RuntimeError(f"Nonfinite simulated state after action {t}")
            success = bool(env._check_success())
            successes.append(success)
            sim_states.append(simstate)
            if success and first_success is None:
                first_success = t
            detail = {}
            if args.task == "OpenDrawer":
                detail = env.drawer.get_door_state(env)
            elif args.task == "OpenStandMixerHead":
                detail = env.stand_mixer.get_state(env)
            details.append({"action_index":action_start + t, "success":success,
                            "conditions":{k:float(v) if isinstance(v, (float,int,np.number)) else str(v)
                                          for k,v in detail.items()}})
            errors.append(float(np.linalg.norm(simstate - states[t+1])) if t+1 < len(states) else None)
            if t % 50 == 0:
                dump(dest / "progress.json", dict(row, phase="action_replay", steps_executed=t+1,
                     success_ever=first_success is not None, elapsed_seconds=time.perf_counter()-started))
        # Final actual state render is supplementary to the frame-aligned video.
        for cam, image in zip(CAMERAS, render()):
            imageio.imwrite(dest / f"final_{cam}.png", image)
        writer.close()
        writer = None
        np.savez_compressed(dest / "replay_trajectory.npz", sim_states_after_action=np.asarray(sim_states),
                    observation_states_before_action=np.asarray(observations), actions_env_order=actions,
                    success_after_action=np.asarray(successes),
                    state_error_l2_mixed_units=np.asarray([np.nan if e is None else e for e in errors]))
        dump(dest / "success_timeline.json", details)
        row.update(steps_executed=len(actions), success_ever=any(successes),
                   success_final=successes[-1],
                   first_success_action=(action_start + first_success) if first_success is not None else None,
                   first_success_seconds=(action_start + first_success + 1)/env.control_freq if first_success is not None else None,
                   max_state_error_l2_mixed_units=max(e for e in errors if e is not None),
                   video=str(dest / "review.mp4"), initial_comparison=str(dest / "initial_comparison.png"))
        # This separate diagnostic cannot turn a failed action replay into success.
        env.sim.set_state_from_flattened(states[-1])
        env.sim.forward()
        if hasattr(env, "update_state"):
            env.update_state()
        row["recorded_last_state_success"] = bool(env._check_success())
        row.update(status="completed", phase="finished", timings=timings,
                   wall_seconds=time.perf_counter()-started)
        dump(dest / "result.json", row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        return 0
    except Exception:
        row.update(status="error", error=traceback.format_exc(),
                   wall_seconds=time.perf_counter()-started, timings=timings)
        dump(dest / "result.json", row)
        traceback.print_exc()
        return 1
    finally:
        if writer is not None:
            writer.close()
        for cap in captures:
            cap.release()
        if env is not None:
            env.close()


def supervise(args):
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "logs").mkdir(exist_ok=True)
    jobs = []
    for task in TASKS:
        episodes = [json.loads(s) for s in (args.dataset / task / "lerobot/meta/episodes.jsonl").read_text().splitlines()]
        jobs.extend((task, int(e["episode_index"])) for e in episodes)
    config = dict(dataset=str(args.dataset), output=str(args.output), gpu=args.gpu,
                  workers=args.workers, total_episodes=len(jobs), seed=0,
                  video_skip=args.video_skip, episode_timeout_seconds=args.timeout,
                  max_attempts=args.attempts, started_at=datetime.datetime.now().astimezone().isoformat(),
                  command=sys.argv, action_source="official LU.get_episode_actions",
                  playback="initial_state_once_then_env.step; no state correction",
                  controller_initialization="reset_to_only; no post-restore controller update/reset",
                  source_action_interval=[args.start_index, args.stop_index],
                  state_error_note="Raw simulator L2 mixes units and is diagnostic, not a physical score.",
                  success_note="success_ever and success_final refer only to continuous action replay")
    dump(args.output / "run_config.json", config)
    dump(args.output / "jobs.json", [dict(task=t, episode_index=e) for t,e in jobs])

    def run_job(job):
        task, ep = job
        result = args.output / "episodes" / task / f"episode_{ep:06d}" / "result.json"
        if result.exists() and json.loads(result.read_text()).get("status") == "completed":
            return json.loads(result.read_text())
        for attempt in range(1, args.attempts+1):
            log = args.output / "logs" / f"{task}_ep{ep:06d}_attempt{attempt}.log"
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--episode", str(ep),
                       "--task", task, "--dataset", str(args.dataset), "--output", str(args.output),
                       "--gpu", str(args.gpu), "--video-skip", str(args.video_skip),
                       "--start-index", str(args.start_index)]
            if args.stop_index is not None:
                command.extend(["--stop-index", str(args.stop_index)])
            with log.open("a") as stream:
                proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = proc.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    code = -1
            if code == 0 and result.exists():
                row = json.loads(result.read_text())
                row.update(attempt=attempt, log=str(log))
                dump(result, row)
                return row
        row = json.loads(result.read_text()) if result.exists() else dict(task=task, episode_index=ep)
        row.update(status="error", attempt=args.attempts, log=str(log), exit_code=code)
        dump(result, row)
        return row

    collected = []
    started = time.perf_counter()
    def summarize(phase):
        fields = ["task", "episode_index", "source_episode_index", "status", "initial_restore_ok",
                  "success_ever", "success_final", "recorded_last_state_success", "steps_executed",
                  "first_success_seconds", "wall_seconds", "video", "initial_comparison", "error"]
        temp = args.output / "results.csv.tmp"
        with temp.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(sorted(collected, key=lambda r:(r["task"],r["episode_index"])))
        temp.replace(args.output / "results.csv")
        by_task = {}
        for task in TASKS:
            rows = [r for r in collected if r["task"]==task]
            completed = [r for r in rows if r["status"]=="completed"]
            by_task[task] = dict(planned=sum(t==task for t,e in jobs), completed=len(completed),
                                errors=sum(r["status"]=="error" for r in rows),
                                success_ever=sum(r.get("success_ever",False) for r in completed),
                                success_final=sum(r.get("success_final",False) for r in completed))
        summary = dict(phase=phase, finished_records=len(collected), planned=len(jobs),
                       elapsed_seconds=time.perf_counter()-started, by_task=by_task)
        dump(args.output / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    # Four real episodes validate the I/O and simulator interfaces, not task success.
    smoke = [(task, min(ep for t,ep in jobs if t==task)) for task in TASKS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(run_job, smoke):
            collected.append(row)
            summarize("smoke")
        if any(row["status"]!="completed" for row in collected):
            summarize("smoke_failed")
            return 1
        remaining = [j for j in jobs if j not in smoke]
        futures = {pool.submit(run_job, j):j for j in remaining}
        for future in concurrent.futures.as_completed(futures):
            collected.append(future.result())
            summarize("running")
    summarize("complete")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--video-skip", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--episode", type=int)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--start-index", type=int, default=0,
                        help="first source action to execute; defaults to the full episode")
    parser.add_argument("--stop-index", type=int, default=None,
                        help="exclusive source action stop index; defaults to the episode end")
    args = parser.parse_args()
    if args.episode is not None:
        # Large scene textures require a per-GPU process limit. New GPU
        # assignments are read before importing any simulator/CUDA libraries.
        # Waiting children read resource changes, without reloading old results.
        import fcntl
        slots = args.output / "gpu_slots"
        slots.mkdir(parents=True, exist_ok=True)
        lease = None
        while lease is None:
            resources_path = args.output / "gpu_resources.json"
            resources = json.loads(resources_path.read_text()) if resources_path.exists() else {
                "gpus": [args.gpu], "simulators_per_gpu": 4}
            for gpu in resources["gpus"]:
                for i in range(resources["simulators_per_gpu"]):
                    handle = (slots / f"gpu{gpu}_{i}.lock").open("a+")
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        lease = handle
                        args.gpu = int(gpu)
                        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
                        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
                        break
                    except BlockingIOError:
                        handle.close()
                if lease is not None:
                    break
            if lease is None:
                time.sleep(1)
        try:
            return replay(args)
        finally:
            lease.close()
    return supervise(args)


if __name__ == "__main__":
    sys.exit(main())
