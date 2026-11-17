"""Read-only execution adapter for the frozen grouped test set. Never invoked by its builder."""

import collections, hashlib, json, time
from pathlib import Path
import numpy as np
from grouped import CAMERAS, STATE_KEYS, dump, digest, restore


def policy_input(env, instruction, size):
    from openpi_client import image_tools

    ob = env._get_observations(force_update=True)
    inputs = {}
    for key, cam in zip(["image", "right_image", "wrist_image"], CAMERAS):
        rgb = np.asarray(env.sim.render(width=256, height=256, camera_name=cam))[::-1].copy()
        inputs["observation/" + key] = image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb, size, size))
    inputs["observation/state"] = np.concatenate(
        [
            np.asarray(ob[k], dtype=np.float32)
            for k in [
                "robot0_base_to_eef_pos",
                "robot0_base_to_eef_quat",
                "robot0_base_pos",
                "robot0_base_quat",
                "robot0_gripper_qpos",
            ]
        ]
    )
    inputs["prompt"] = instruction
    return inputs


def native_action(env, prediction):
    # Follow the official gym wrapper, including its 0.5 gripper/mode thresholds.
    from robocasa.wrappers.gym_wrapper import PandaOmronKeyConverter
    from robocasa.utils.env_utils import convert_action
    from robosuite.controllers.composite.composite_controller import HybridMobileBase

    mapped = PandaOmronKeyConverter.unmap_action(convert_action(prediction))
    parts = []
    for robot in env.robots:
        cc = robot.composite_controller
        prefix = robot.robot_model.naming_prefix
        action = np.zeros(cc.action_limits[0].shape)
        for name in cc.part_controllers:
            a, b = cc._action_split_indexes[name]
            action[a:b] = mapped.pop(prefix + name)
        if isinstance(cc, HybridMobileBase):
            action[-1] = mapped.pop(prefix + "base_mode")
        parts.append(action)
    if mapped:
        raise RuntimeError("Unprocessed official action fields")
    return np.concatenate(parts)


def load_case(dataset, entry):
    dataset = Path(dataset)
    setting = json.loads((dataset / entry["configuration_file"]).read_text())
    anchor = json.loads((dataset / setting["anchor"]).read_text())
    anchor["directory"] = str(dataset / anchor["directory"])
    return setting, anchor, dataset / setting["state_file"]


def evaluate(config, output, model, port, *, dataset=None, gpu=None, shard_index=0, shard_count=1, max_episodes=None):
    import os, sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "robocasa_augmentation"))
    from gpu_routing import egl_device_for_uuid

    # Caller selects the rendering GPU; never attach to another policy by default.
    import subprocess

    if gpu is not None:
        uuid = subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"], text=True
        ).strip()
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_device_for_uuid(uuid))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    from env import make_env
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    import imageio.v2 as imageio

    checkpoint = config.get("models", {}).get(model, {}).get("checkpoint")
    policy_config = config.get("models", {}).get(model, {}).get("policy_config")
    if not checkpoint or not policy_config:
        raise RuntimeError("Set checkpoint and policy_config for " + model)
    dataset = Path(dataset) if dataset is not None else output
    manifest = json.loads((dataset / "manifest.json").read_text())
    client = WebsocketClientPolicy("127.0.0.1", port)
    meta = client.get_server_metadata()
    if (
        not meta.get("explicit_inference_seed")
        or str(Path(meta["checkpoint"]).resolve()) != str(Path(checkpoint).resolve())
        or meta.get("policy_config") != policy_config
    ):
        raise RuntimeError("Independent seeded policy service/checkpoint mismatch")
    entries = [
        entry
        for index, entry in enumerate(manifest["cases"])
        if index % shard_count == shard_index and entry["accepted"]
    ]
    if max_episodes is not None:
        entries = entries[:max_episodes]
    for entry in entries:
        case = entry["case"]
        dest = output / "results" / model / case["task"] / f"scene_{case['scene_index']:02d}" / case["id"]
        record = dest / "result.json"
        if record.exists() and json.loads(record.read_text()).get("status") == "completed":
            continue
        setting, anchor, state_file = load_case(dataset, entry)
        if hashlib.sha256(state_file.read_bytes()).hexdigest() != entry["state_sha256"]:
            raise RuntimeError("Frozen state hash mismatch")
        env = None
        dest.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        try:
            env = make_env(
                case["task"],
                anchor["scene"],
                anchor["scene"]["seed"],
                int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "-1")),
            )
            with np.load(state_file) as z:
                arrays = {k: z[k].copy() for k in z.files}
            restore(env, anchor, arrays, reload_model=True)
            obs = env._get_observations(force_update=True)
            state = np.concatenate([np.asarray(obs[k]) for k in STATE_KEYS])
            if not np.allclose(state, setting["observation_state"], atol=1e-7, rtol=0):
                raise RuntimeError("Restored first-frame State differs from frozen sample")
            actions = []
            chunks = collections.deque()
            chunk_id = 0
            success = False
            with imageio.get_writer(str(dest / "review.mp4"), fps=20, codec="libx264", pixelformat="yuv420p") as writer:
                for step in range(get_task_horizon(case["task"])):
                    if not chunks:
                        inputs = policy_input(env, anchor["instruction"], config["inference"]["resize_size"])
                        inputs["__inference_seed"] = int(
                            digest([case["inference_seed"], case["task"], case["scene_index"], case["id"], chunk_id])[
                                :8
                            ],
                            16,
                        )
                        chunk = np.asarray(client.infer(inputs)["actions"])
                        chunk_id += 1
                        if chunk.ndim != 2 or chunk.shape[1] != 12 or not np.isfinite(chunk).all():
                            raise RuntimeError("Invalid decoded 12D action chunk")
                        np.save(dest / f"chunk_{chunk_id:05d}.npy", chunk)
                        chunks.extend(chunk[: config["inference"]["replan_steps"]])
                    prediction = chunks.popleft()
                    action = native_action(env, prediction)
                    env.step(action)
                    actions.append(prediction)
                    success = bool(env._check_success())
                    if step % 2 == 0 or success:
                        writer.append_data(
                            np.asarray(env.sim.render(width=256, height=256, camera_name=CAMERAS[0]))[::-1]
                        )
                    if success:
                        break
            np.savez_compressed(dest / "actions.npz", actions=np.asarray(actions))
            dump(
                record,
                dict(
                    **case,
                    status="completed",
                    success=success,
                    steps=len(actions),
                    termination="success" if success else "timeout",
                    total_seconds=time.perf_counter() - start,
                    actual_delta=setting["actual_delta"],
                    checkpoint=checkpoint,
                    state_sha256=entry["state_sha256"],
                ),
            )
        except Exception as ex:
            dump(record, dict(**case, status="error", error=repr(ex)))
        finally:
            if env is not None:
                env.close()


def report(config, output):
    import pandas as pd

    for model in config.get("models", {}):
        rows = [json.loads(p.read_text()) for p in (output / "results" / model).glob("*/*/*/result.json")]
        if not rows:
            continue
        d = pd.DataFrame(rows)
        d.to_csv(output / f"{model}_results.csv", index=False)
        completed = d[d.status == "completed"]
        for name, keys in [
            ("task", ["task"]),
            ("condition", ["task", "family", "level"]),
            ("scene", ["task", "scene_index"]),
        ]:
            if len(completed):
                completed.groupby(keys, dropna=False).success.agg(["sum", "count", "mean"]).to_csv(
                    output / f"{model}_{name}_summary.csv"
                )
        dump(
            output / f"{model}_summary.json",
            dict(
                planned=1460,
                completed=len(completed),
                successes=int(completed.success.sum()) if len(completed) else 0,
                success_rate=float(completed.success.mean()) if len(completed) else None,
                errors=int((d.status == "error").sum()),
                task_macro_success_rate=float(completed.groupby("task").success.mean().mean())
                if len(completed)
                else None,
            ),
        )
