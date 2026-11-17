"""Fixed-scene initial-configuration benchmark. No simulator is imported at plan time."""
import argparse
import collections
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=lambda x: np.asarray(x).tolist()))
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=lambda x: np.asarray(x).tolist()).encode()).hexdigest()


def start_hash(config):
    return digest({k: v for k, v in config.items() if k not in ("checkpoint", "policy_config", "inference", "repeats", "baseline_repeats")})


def comparable_metadata(field, value):
    value = deepcopy(value)
    if field == "object_cfgs":
        for cfg in value:
            info = cfg.get("info", {})
            # Official deterministic playback replaces the sampling-category query
            # with a concrete asset path. Compare identity/geometry, not that query.
            info.pop("groups", None)
            if "groups_containing_sampled_obj" in info:
                info["groups_containing_sampled_obj"] = sorted(info["groups_containing_sampled_obj"])
    return value


def cases(config):
    for task in config["tasks"]:
        yield dict(task=task, id="baseline", family="baseline", axis=None, value=0)
        if config["sampling"]["mode"] == "stratified_random":
            groups = [("body", axis, bins) for axis, bins in config["sampling"]["body_magnitude_bins"].items()]
            groups += [("arm", j, config["sampling"]["arm_magnitude_bins_deg"])
                       for j in config["arm_perturbations"]["joint_indices"]]
            for family, axis, bins in groups:
                for level, (low, high) in zip(("small", "medium", "large"), bins, strict=True):
                    if not 0 <= low < high:
                        raise ValueError(f"Invalid magnitude bin: {low}, {high}")
                    for sign in (-1, 1):
                        for sample in range(config["sampling"]["samples_per_stratum"]):
                            yield dict(task=task, id=f"{family}_{axis}_{level}_{'neg' if sign < 0 else 'pos'}_{sample}",
                                       family=family, axis=axis, value=None, level=level, sign=sign,
                                       magnitude_bin=[low, high], sample_index=sample)
            continue
        if config["sampling"]["mode"] != "fixed_sweep":
            raise ValueError("sampling.mode must be stratified_random or fixed_sweep")
        for axis, values in config["body_perturbations"].items():
            for i, value in enumerate(values):
                yield dict(task=task, id=f"body_{axis}_{i}", family="body", axis=axis, value=value)
        for joint in config["arm_perturbations"]["joint_indices"]:
            for i, value in enumerate(config["arm_perturbations"]["delta_deg"]):
                yield dict(task=task, id=f"arm_j{joint}_{i}", family="arm", axis=joint, value=value)


def make_env(task, scene):
    import gymnasium as gym
    import robocasa  # noqa: F401: registers environments
    # split='pretrain' overrides explicit layout/style in upstream create_env.
    # Use split=None plus explicit object split to genuinely fix the kitchen.
    np.random.seed(scene["seed"])
    return gym.make(
        f"robocasa/{task}", split=None, seed=scene["seed"],
        layout_ids=[scene["layout_id"]], style_ids=[scene["style_id"]],
        obj_instance_split=scene["object_split"], randomize_cameras=False,
    )


def joint_mapping(env):
    robot = env.robots[0]
    arm = list(robot.robot_model.arm_joints)
    base = list(robot.robot_model.base_joints)
    torso = list(robot.robot_model.torso_joints)
    if len(arm) != 7 or len(torso) != 1:
        raise RuntimeError(f"Unexpected PandaOmron joint mapping: {arm}, {torso}")
    body = next(n for n in env.sim.model.body_names if n == "mobilebase0_base")
    return {"arm": arm, "base": base, "torso": torso, "base_body": body}


def frozen_model(sim):
    # Local mounting extrinsics stay fixed; robot-mounted cameras move WITH the robot.
    return {key: np.asarray(getattr(sim.model, key)).copy() for key in
            ("body_pos", "body_quat", "cam_pos", "cam_quat", "cam_fovy")}


def contacts(env):
    sim = env.sim
    records = {}
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        names = sorted([sim.model.geom_id2name(c.geom1) or f"geom{c.geom1}",
                        sim.model.geom_id2name(c.geom2) or f"geom{c.geom2}"])
        if not any(n.startswith(("robot0_", "mobilebase0_", "gripper0_")) for n in names):
            continue
        key = "|".join(names)
        records[key] = min(records.get(key, 0.0), float(c.dist))
    return records


def official_collision(env):
    from robocasa.utils.env_utils import detect_robot_collision
    # Upstream caches geometry IDs. Rebuild them for the current loaded XML.
    env.robot_geom_ids = None
    return bool(detect_robot_collision(env))


def prepare(wrapper, config, task, directory):
    env = wrapper.unwrapped.env
    obs, _ = wrapper.reset(seed=config["scene"]["seed"])
    meta = env.get_ep_meta()
    scene = config["scene"]
    if (meta["layout_id"], meta["style_id"]) != (scene["layout_id"], scene["style_id"]):
        raise RuntimeError("Requested fixed kitchen was not respected")
    mapping = joint_mapping(env)
    directory.mkdir(parents=True, exist_ok=False)
    xml = env.sim.model.get_xml()
    (directory / "model.xml").write_text(xml)
    np.savez_compressed(directory / "state.npz", state=env.sim.get_state().flatten(),
                        ctrl=env.sim.data.ctrl.copy(), **frozen_model(env.sim))
    dump(directory / "anchor.json", {
        "task": task, "config_hash": digest(config), "start_hash": start_hash(config), "scene": scene,
        "ep_meta": meta, "mapping": mapping,
        "instruction": obs["annotation.human.task_description"],
        "rng_state": env.rng.bit_generator.state,
        "numpy_rng_state": list(np.random.get_state()),
        "xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "initial_contacts": contacts(env), "initial_official_collision": official_collision(env),
        "control_freq": env.control_freq,
    })
    save_images(directory, obs)


def save_images(directory, obs):
    import imageio.v2 as imageio
    for camera in ("agentview_left", "agentview_right", "eye_in_hand"):
        imageio.imwrite(directory / f"{camera}.png", obs[f"video.robot0_{camera}"])


def restore(wrapper, anchor, saved):
    env = wrapper.unwrapped.env
    # Runtime body poses / camera mounts are not part of MjSimState, and XML
    # export can retain their pre-initialization values. Restore them explicitly.
    for key in ("body_pos", "body_quat", "cam_pos", "cam_quat", "cam_fovy"):
        getattr(env.sim.model, key)[:] = saved[key]
    # Rebuild controller internals/buffers without regenerating scene or settling physics.
    for robot in env.robots:
        robot.reset(deterministic=True)
    env.sim.set_state_from_flattened(saved["state"])
    env.sim.data.ctrl[:] = saved["ctrl"]
    env.sim.forward()
    env.set_ep_meta(deepcopy(anchor["ep_meta"]))
    env.cur_time, env.timestep, env.done = 0, 0, False
    env._obs_cache = {}
    env._reset_observables()
    env.rng.bit_generator.state = anchor["rng_state"]
    s = anchor["numpy_rng_state"]
    np.random.set_state((s[0], np.array(s[1], dtype=np.uint32), int(s[2]), int(s[3]), float(s[4])))
    for robot in env.robots:
        robot.base_pos = env.sim.data.get_body_xpos(robot.robot_model.root_body).copy()
        robot.base_ori = env.sim.data.get_body_xmat(robot.robot_model.root_body).copy()
        cc = robot.composite_controller
        cc.update_state()
        for controller in cc.part_controllers.values():
            controller.update(force=True)
        cc.reset()


def apply_case(env, mapping, case):
    sim = env.sim
    old = sim.data.qpos.copy()
    arm_ids = [sim.model.get_joint_qpos_addr(j) for j in mapping["arm"]]
    base_ids = [sim.model.get_joint_qpos_addr(j) for j in mapping["base"]]
    torso_id = sim.model.get_joint_qpos_addr(mapping["torso"][0])
    axis, value = case["axis"], case["value"]
    before_pos = sim.data.get_body_xpos(mapping["base_body"]).copy()
    before_rot = sim.data.get_body_xmat(mapping["base_body"]).copy().reshape(3, 3)
    if case["family"] == "arm":
        sim.data.qpos[arm_ids[int(axis)]] += np.deg2rad(value)
    elif case["family"] == "body":
        if axis in ("dx_m", "dy_m"):
            names = [next(j for j in mapping["base"] if j.endswith(s)) for s in
                     ("joint_mobile_forward", "joint_mobile_side")]
            ids = [sim.model.get_joint_qpos_addr(j) for j in names]
            # Determine true world translation axes rather than assuming world XY = joint XY.
            jac = []
            for address in ids:
                sim.data.qpos[address] += 0.001
                sim.forward()
                jac.append((sim.data.get_body_xpos(mapping["base_body"])[:2] - before_pos[:2]) / 0.001)
                sim.data.qpos[:] = old
                sim.forward()
            desired = before_rot[:2, 0 if axis == "dx_m" else 1] * value
            sim.data.qpos[ids] += np.linalg.solve(np.array(jac).T, desired)
        elif axis == "dyaw_deg":
            name = next(j for j in mapping["base"] if j.endswith("joint_mobile_yaw"))
            sim.data.qpos[sim.model.get_joint_qpos_addr(name)] += np.deg2rad(value)
        elif axis == "dh_m":
            sim.data.qpos[torso_id] += value
        else:
            raise ValueError(axis)
    sim.forward()
    if case["family"] == "body":
        assert np.array_equal(sim.data.qpos[arm_ids], old[arm_ids])
    if case["family"] == "arm":
        assert np.array_equal(sim.data.qpos[base_ids + [torso_id]], old[base_ids + [torso_id]])
    changed = set(np.flatnonzero(sim.data.qpos != old))
    allowed = set(arm_ids if case["family"] == "arm" else base_ids + [torso_id])
    assert changed <= allowed, "Non-robot state changed"
    limits = []
    for name in mapping["arm"] + mapping["torso"]:
        j = sim.model.joint_name2id(name)
        q = float(sim.data.qpos[sim.model.get_joint_qpos_addr(name)])
        if sim.model.jnt_limited[j] and not sim.model.jnt_range[j, 0] <= q <= sim.model.jnt_range[j, 1]:
            limits.append({"joint": name, "qpos": q, "range": sim.model.jnt_range[j].tolist()})
    return {"status": "invalid_joint_limit" if limits else "ready", "joint_limit_violations": limits,
            "actual_qpos_delta": (sim.data.qpos - old).tolist(),
            "base_world_position": sim.data.get_body_xpos(mapping["base_body"]).tolist(),
            "base_world_rotation": sim.data.get_body_xmat(mapping["base_body"]).tolist(),
            "torso_qpos": float(sim.data.qpos[torso_id]),
            "arm_qpos": sim.data.qpos[arm_ids].tolist()}


def observation(wrapper):
    env = wrapper.unwrapped.env
    # Refresh sensors at the edited state; no env.step / hidden stabilization time.
    return wrapper.unwrapped.get_observation(env._get_observations(force_update=True))


def check_candidate(wrapper, anchor, saved, case):
    env = wrapper.unwrapped.env
    restore(wrapper, anchor, saved)
    outcome = apply_case(env, anchor["mapping"], case)
    for key, value in frozen_model(env.sim).items():
        if not np.array_equal(value, saved[key]):
            raise RuntimeError(f"Static model/camera mounting changed: {key}")
    collision = official_collision(env)
    outcome.update(contacts=contacts(env), official_collision=collision,
                   collision_check="robocasa.utils.env_utils.detect_robot_collision")
    if collision and outcome["status"] == "ready":
        outcome["status"] = "invalid_official_collision"
    return outcome


def sample_case(wrapper, anchor, saved, slot, config, directory):
    """One shared attempt budget per axis/direction/level; save distinct valid starts."""
    record = directory / "configuration.json"
    if record.exists():
        return json.loads(record.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    random_slot = slot["value"] is None
    group = slot["id"].rsplit("_", 1)[0] if random_slot else slot["id"]
    target_index = slot["sample_index"] if random_slot else 0
    seed = int(digest([config["sampling"]["seed"], slot["task"], group])[:16], 16)
    attempts_path = directory.parents[2] / "sampling_attempts" / slot["task"] / group / "attempts.jsonl"
    attempts_path.parent.mkdir(parents=True, exist_ok=True)
    previous = [json.loads(line) for line in attempts_path.read_text().splitlines()] if attempts_path.exists() else []
    reasons = collections.Counter(x["outcome"]["status"] for x in previous)
    limit = config["sampling"]["max_attempts_per_stratum"] if random_slot else 1
    passes = [x for x in previous if x["outcome"]["status"] == "ready"]
    accepted = passes[target_index] if len(passes) > target_index else None
    for attempt in range(len(previous), limit) if accepted is None else ():
        candidate = dict(slot)
        if candidate["value"] is None:
            candidate.update(id=f"{group}_{len(passes)}", sample_index=len(passes))
            rng = np.random.default_rng(np.random.SeedSequence([seed, attempt]))
            low, high = slot["magnitude_bin"]
            candidate["value"] = slot["sign"] * float(rng.uniform(np.nextafter(float(low), float(high)), high))
        try:
            if any(abs(candidate["value"] - x["case"]["value"]) < 1e-12 for x in passes):
                outcome = dict(status="invalid_duplicate")
            else:
                outcome = check_candidate(wrapper, anchor, saved, candidate)
        except Exception as exc:
            import traceback
            outcome = dict(status="error", error=repr(exc), traceback=traceback.format_exc())
        event = dict(attempt=attempt, case=candidate, outcome=outcome)
        with attempts_path.open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        previous.append(event)
        reasons[outcome["status"]] += 1
        if outcome["status"] == "ready":
            passes.append(event)
            if len(passes) > target_index:
                accepted = event
                break
        if outcome["status"] == "error":
            break
    if accepted is not None:
        outcome = check_candidate(wrapper, anchor, saved, accepted["case"])
        if outcome["status"] != "ready":
            raise RuntimeError("Accepted sample could not be reproduced")
        obs = observation(wrapper)
        save_images(directory, obs)
        env = wrapper.unwrapped.env
        np.savez_compressed(directory / "state.npz", qpos=env.sim.data.qpos.copy(),
                            qvel=env.sim.data.qvel.copy(), ctrl=env.sim.data.ctrl.copy())
        result = dict(case=accepted["case"], accepted=True, outcome=outcome)
    else:
        result = dict(case=slot, accepted=False,
                      outcome={"status": "sampling_error" if reasons["error"] else "sampling_exhausted"})
    result.update(sampling_seed=seed, attempts=len(previous), status_counts=dict(reasons),
                  sampling_group=group, attempts_log=str(attempts_path),
                  config_hash=digest(config), start_hash=start_hash(config), anchor_xml_sha256=anchor["xml_sha256"])
    dump(record, result)
    return result


def summarize_samples(output):
    records = [json.loads(p.read_text()) for p in sorted((output / "configurations").glob("*/*/configuration.json"))]
    if not records:
        return
    with (output / "sampling_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["task", "id", "family", "axis", "level", "sign", "value", "magnitude_bin",
                                                    "accepted", "attempts", "sampling_seed", "status_counts"])
        writer.writeheader()
        for record in records:
            row = {k: record["case"].get(k) for k in ("task", "id", "family", "axis", "level", "sign", "value", "magnitude_bin")}
            row.update({k: record[k] for k in ("accepted", "attempts", "sampling_seed", "status_counts")})
            writer.writerow(row)
    with (output / "sampling_groups.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["task", "group", "attempts", "valid_configurations", "status_counts"])
        writer.writeheader()
        for path in sorted((output / "sampling_attempts").glob("*/*/attempts.jsonl")):
            events = [json.loads(line) for line in path.read_text().splitlines()]
            counts = collections.Counter(x["outcome"]["status"] for x in events)
            writer.writerow(dict(task=path.parent.parent.name, group=path.parent.name,
                                 attempts=len(events), valid_configurations=counts["ready"], status_counts=dict(counts)))


def policy_input(obs, size):
    from openpi_client import image_tools
    item = {}
    for key, camera in (("image", "agentview_left"), ("right_image", "agentview_right"), ("wrist_image", "eye_in_hand")):
        item[f"observation/{key}"] = image_tools.convert_to_uint8(image_tools.resize_with_pad(
            np.ascontiguousarray(obs[f"video.robot0_{camera}"]), size, size))
    item["observation/state"] = np.concatenate([obs[f"state.{k}"] for k in (
        "end_effector_position_relative", "end_effector_rotation_relative", "base_position", "base_rotation", "gripper_qpos")])
    item["prompt"] = obs["annotation.human.task_description"]
    return item


def evaluate(wrapper, anchor, saved, case, config, client, destination, sampled_state):
    import imageio.v2 as imageio
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from robocasa.utils.env_utils import convert_action
    env = wrapper.unwrapped.env
    started = time.perf_counter()
    outcome = check_candidate(wrapper, anchor, saved, case)
    with np.load(sampled_state) as reference:
        if not np.allclose(env.sim.data.qpos, reference["qpos"], rtol=0, atol=1e-12):
            raise RuntimeError("Restored candidate differs from the saved sampled configuration")
    obs = observation(wrapper)
    assert obs["annotation.human.task_description"] == anchor["instruction"]
    save_images(destination, obs)
    np.savez_compressed(destination / "initial_state.npz", qpos=env.sim.data.qpos.copy(),
                        qvel=env.sim.data.qvel.copy(), ctrl=env.sim.data.ctrl.copy(),
                        policy_state=policy_input(obs, config["inference"]["resize_size"])["observation/state"])
    if outcome["status"] != "ready":
        return outcome
    # Reset control goals at the perturbed pose without changing physical qpos.
    for robot in env.robots:
        robot.base_pos = env.sim.data.get_body_xpos(robot.robot_model.root_body).copy()
        robot.base_ori = env.sim.data.get_body_xmat(robot.robot_model.root_body).copy()
        robot.composite_controller.update_state()
        for controller in robot.composite_controller.part_controllers.values():
            controller.update(force=True)
        robot.composite_controller.reset()
    horizon = get_task_horizon(case["task"])
    plan, actions, states = collections.deque(), [], []
    success = False
    inference_seconds = 0.0
    with imageio.get_writer(destination / "review.mp4", fps=20,
                            codec="libx264", pixelformat="yuv420p") as writer:
        writer.append_data(wrapper.render())
        for step in range(horizon):
            inputs = policy_input(obs, config["inference"]["resize_size"])
            states.append(inputs["observation/state"])
            if not plan:
                t = time.perf_counter()
                chunk = np.asarray(client.infer(inputs)["actions"])
                inference_seconds += time.perf_counter() - t
                k = config["inference"]["replan_steps"]
                if chunk.ndim != 2 or chunk.shape[0] < k or chunk.shape[1] != 12 or not np.isfinite(chunk).all():
                    raise RuntimeError(f"Invalid policy chunk: {chunk.shape}")
                np.save(destination / f"chunk_{step:05d}.npy", chunk)
                plan.extend(chunk[:k])
            action = plan.popleft()
            actions.append(action.copy())
            obs, _, _, _, info = wrapper.step(convert_action(action))
            success = bool(info["success"])
            if step % 2 == 0 or success or step == horizon - 1:
                writer.append_data(wrapper.render())
            if success:
                break
    np.savez_compressed(destination / "trajectory.npz", actions=np.asarray(actions), policy_states=np.asarray(states))
    outcome.update(status="completed", success=success, steps=len(actions), horizon=horizon,
                   termination="success" if success else "timeout", inference_seconds=inference_seconds,
                   total_seconds=time.perf_counter() - started)
    return outcome


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "benchmark.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--starts-root", type=Path, help="Read an existing anchor/configuration pool for evaluate only")
    parser.add_argument("--mode", choices=["plan", "prepare", "sample", "evaluate"], default="plan")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18281)
    parser.add_argument("--task", choices=["PickPlaceSinkToCounter", "OpenDrawer", "OpenStandMixerHead", "PickPlaceCounterToCabinet"])
    parser.add_argument("--case", help="Run only this case ID, e.g. baseline")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    if args.starts_root is not None and args.mode != "evaluate":
        parser.error("--starts-root is only supported for evaluate")
    starts_root = args.starts_root or args.output
    config = json.loads(args.config.read_text())
    selected = [c for c in cases(config) if (not args.task or c["task"] == args.task) and (not args.case or c["id"] == args.case)]
    # A changed configuration must use a new output root: prevents mixing cached starts/results.
    marker = args.output / "config.json"
    if marker.exists() and digest(json.loads(marker.read_text())) != digest(config):
        raise RuntimeError("Output directory contains a different configuration; choose a new output")
    dump(marker, config)
    dump(args.output / "plan.json", selected)
    if args.mode == "plan":
        count = sum(config['baseline_repeats'] if c['family'] == 'baseline' else config['repeats'] for c in selected)
        print(f"Planned {len(selected)} configuration slots, up to {count} rollouts. No simulation started.")
        return
    client = None
    if args.mode == "evaluate":
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        client = WebsocketClientPolicy(args.host, args.port)
        dump(args.output / "service.json", {"host": args.host, "port": args.port,
             "declared_checkpoint": config["checkpoint"],
             "server_metadata": client.get_server_metadata(),
             "policy_rng": "service-owned stream; no per-trial sampling-seed API"})
    for task in dict.fromkeys(c["task"] for c in selected):
        wrapper = make_env(task, config["scene"])
        try:
            directory = starts_root / "anchors" / task
            if args.mode == "prepare":
                if directory.exists():
                    raise RuntimeError(f"Anchor already exists, refusing to overwrite: {directory}")
                prepare(wrapper, config, task, directory)
                continue
            anchor = json.loads((directory / "anchor.json").read_text())
            if anchor["start_hash"] != start_hash(config):
                raise RuntimeError("Anchor configuration mismatch")
            if hashlib.sha256((directory / "model.xml").read_bytes()).hexdigest() != anchor["xml_sha256"]:
                raise RuntimeError("Anchor XML checksum mismatch")
            env = wrapper.unwrapped.env
            # Recreate the same Python-side task/object models as anchor preparation,
            # not merely the saved MuJoCo geometry. This reset happens once per task.
            env.set_ep_meta(deepcopy(anchor["ep_meta"]))
            wrapper.reset(seed=config["scene"]["seed"])
            recreated = env.get_ep_meta()
            for field in ("layout_id", "style_id", "object_cfgs", "fixture_refs"):
                if digest(comparable_metadata(field, recreated[field])) != digest(comparable_metadata(field, anchor["ep_meta"][field])):
                    dump(args.output / "diagnostics" / f"{task}_{field}_mismatch.json",
                         {"expected": anchor["ep_meta"][field], "recreated": recreated[field]})
                    raise RuntimeError(f"Anchor task/object reconstruction mismatch: {field}")
            env.set_ep_meta(deepcopy(anchor["ep_meta"]))
            env.reset_from_xml_string((directory / "model.xml").read_text())
            if joint_mapping(env) != anchor["mapping"]:
                raise RuntimeError("Anchor joint mapping mismatch")
            with np.load(directory / "state.npz") as saved:
                for case in (c for c in selected if c["task"] == task):
                    config_dir = starts_root / "configurations" / task / case["id"]
                    if args.mode == "sample":
                        record = sample_case(wrapper, anchor, saved, case, config, config_dir)
                        summarize_samples(args.output)
                        print(task, case["id"], record["accepted"], record["attempts"], flush=True)
                        if record["outcome"]["status"] == "sampling_error":
                            raise RuntimeError(f"Sampler infrastructure error: {config_dir}; see attempts log")
                        continue
                    record = json.loads((config_dir / "configuration.json").read_text())
                    if record["start_hash"] != start_hash(config) or record["anchor_xml_sha256"] != anchor["xml_sha256"]:
                        raise RuntimeError("Saved sampled configuration does not match this run")
                    if not record["accepted"]:
                        print(task, case["id"], record["outcome"]["status"], "not a policy failure", flush=True)
                        continue
                    case = record["case"]
                    for repeat in range(config["baseline_repeats"] if case["family"] == "baseline" else config["repeats"]):
                        dest = args.output / "trials" / task / f'{case["id"]}_r{repeat}'
                        result = dest / "result.json"
                        if result.exists() and not (args.retry_errors and json.loads(result.read_text())["status"] == "error"):
                            continue
                        dest.mkdir(parents=True, exist_ok=True)
                        for connection_attempt in range(2):
                            try:
                                outcome = evaluate(wrapper, anchor, saved, case, config, client, dest, config_dir / "state.npz")
                                break
                            except Exception as exc:
                                import traceback
                                import socket
                                from websockets.exceptions import ConnectionClosed
                                if connection_attempt == 0 and isinstance(exc, (ConnectionClosed, OSError)):
                                    print(f"Connection lost for {task} {case['id']}; reconnecting and restarting this rollout", flush=True)
                                    try:
                                        with socket.create_connection((args.host, args.port), timeout=5):
                                            pass
                                        client = WebsocketClientPolicy(args.host, args.port)
                                        continue
                                    except Exception as reconnect_exc:
                                        outcome = {"status": "error", "error": repr(reconnect_exc), "traceback": traceback.format_exc()}
                                        break
                                outcome = {"status": "error", "error": repr(exc), "traceback": traceback.format_exc()}
                                break
                        dump(result, {**case, "repeat": repeat, "scene": anchor["scene"],
                                      "sampled_configuration": str(config_dir / "configuration.json"),
                                      "checkpoint": config["checkpoint"], **outcome})
                        print(task, case["id"], repeat, outcome["status"], flush=True)
        finally:
            wrapper.close()
    results = [json.loads(f.read_text()) for f in sorted((args.output / "trials").glob("*/*/result.json"))]
    if results:
        keys = ["task", "id", "family", "axis", "level", "sign", "value", "repeat", "status", "official_collision", "success", "steps", "termination", "total_seconds"]
        with (args.output / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        completed = [r for r in results if r["status"] == "completed"]
        dump(args.output / "summary.json", {"recorded": len(results), "completed": len(completed),
             "successes": sum(r["success"] for r in completed),
             "status_counts": dict(collections.Counter(r["status"] for r in results)),
             "note": "Invalid and error cases are separate from completed rollouts."})


if __name__ == "__main__":
    main()
