"""Export RoboCasa camera poses and robot segmentation masks for one episode.

Run this helper with the RoboCasa environment, not the SHARP/PyTorch environment.
It restores the episode XML and first state, computes the native task reset
anchor, and saves only calibration/masks; the SHARP process consumes this file.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from examples.robocasa_spatial.generate_teacher_cache import (
    CAMERA_KEYS,
    camera_calibration,
    robot_geom_ids,
    robot_mask,
)


PROJECT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = Path("/data4/xuchenfei/robocasa365/output/anchor_visual_audit")
DATA_ROOT = Path("/data4/xuchenfei/robocasa365/third_party/robocasa/datasets/v1.0/pretrain/atomic")
TASKS = ("PickPlaceSinkToCounter", "OpenDrawer", "OpenStandMixerHead", "PickPlaceCounterToCabinet")


def _audit_module():
    path = AUDIT_ROOT / "build_audit.py"
    spec = importlib.util.spec_from_file_location("robocasa_anchor_audit_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import anchor audit functions from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--episode", required=True, type=int)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    a = p.parse_args()
    import robocasa  # noqa: F401 - register RoboCasa tasks
    import robosuite as suite

    audit = _audit_module()
    root = next((a.data_root / a.task).glob("*/lerobot"))
    episode_dir = root / "extras" / f"episode_{a.episode:06d}"
    meta = json.loads((episode_dir / "ep_meta.json").read_text())
    dataset_meta = json.loads((root / "extras/dataset_meta.json").read_text())
    env_kwargs = dict(dataset_meta["env_args"]["env_kwargs"])
    camera_names = list(CAMERA_KEYS)
    env_kwargs.update(
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        camera_names=camera_names,
        camera_widths=256,
        camera_heights=256,
        camera_depths=False,
        render_gpu_device_id=-1,
        seed=0,
    )
    xml_path = episode_dir / "model.xml.gz"
    state0 = np.load(episode_dir / "states.npz", allow_pickle=False)["states"][0]
    parquet_path = root / "data" / f"chunk-{a.episode//1000:03d}" / f"episode_{a.episode:06d}.parquet"
    observation_state0 = np.asarray(
        pd.read_parquet(parquet_path, columns=["observation.state"])["observation.state"].iloc[0],
        dtype=np.float64,
    )
    if observation_state0.shape != (16,):
        raise RuntimeError(f"expected 16D LeRobot observation.state, got {observation_state0.shape}")
    env = suite.make(**env_kwargs)
    try:
        env.set_ep_meta(meta)
        env.reset()
        xml_text = gzip.open(xml_path, "rt").read()
        xml_root, matched_fixtures = audit.set_xml_fixture_cfg(env, xml_path)
        anchor_xyz, anchor_rpy = audit.get_anchor(a.task, env, meta, xml_root)
        # Rebuild from the exact recorded model/state before measuring cameras.
        env.reset_from_xml_string(env.edit_model_xml(xml_text))
        env.sim.reset()
        env.sim.set_state_from_flattened(state0)
        env.sim.forward()
        geom_ids = robot_geom_ids(env.sim)
        h = w = 256
        arrays: dict[str, np.ndarray] = {}
        cam_meta = {}
        for cam in camera_names:
            k, t_world_src, fovy = camera_calibration(env.sim, cam, h, w)
            arrays[f"K_{cam}"] = k
            arrays[f"T_world_source_{cam}"] = t_world_src
            arrays[f"robot_mask_source_{cam}"] = robot_mask(env.sim, cam, w, h, geom_ids)
            cam_meta[cam] = {"fovy_deg": fovy, "source_robot_mask_fraction": float(arrays[f"robot_mask_source_{cam}"].mean())}

        # Match RoboCasa's own world->local base joint convention; preserve all
        # arm/gripper qpos and all non-robot scene state.
        anchor_pose_check = audit.set_base_pose(
            env, np.asarray(anchor_xyz[:2], float), float(anchor_rpy[2]), anchor_rpy
        )
        for cam in camera_names:
            _, t_world_ref, _ = camera_calibration(env.sim, cam, h, w)
            arrays[f"T_world_anchor_{cam}"] = t_world_ref
            arrays[f"robot_mask_anchor_{cam}"] = robot_mask(env.sim, cam, w, h, geom_ids)
            cam_meta[cam]["anchor_robot_mask_fraction"] = float(arrays[f"robot_mask_anchor_{cam}"].mean())

        # Compare computed native reset anchor with the already rendered audit row.
        csv_path = AUDIT_ROOT / "anchor_candidates.csv"
        row = next(
            (r for r in __import__("csv").DictReader(csv_path.open(newline=""))
             if r["task"] == a.task and int(r["episode_index"]) == a.episode),
            None,
        )
        candidate_consistency = None
        if row and row.get("anchor_x"):
            candidate_consistency = {
                "xy_error_m": float(np.linalg.norm(np.asarray(anchor_xyz[:2]) - [float(row["anchor_x"]), float(row["anchor_y"])])),
                "yaw_error_deg": float(np.rad2deg((float(anchor_rpy[2]) - np.deg2rad(float(row["anchor_yaw_deg"])) + np.pi) % (2*np.pi) - np.pi)),
            }
        # states.npz stores MuJoCo's flattened simulator state, not the 16D
        # LeRobot observation. Use the actual recorded state row for pose labels.
        arrays["human_base_xyz_world"] = observation_state0[:3]
        arrays["human_base_quaternion_xyzw"] = observation_state0[3:7]
        arrays["human_base_root_xyz_world_from_metadata"] = np.asarray(
            meta["init_robot_base_pos"], dtype=np.float64
        )
        arrays["human_base_root_rpy_world_from_metadata"] = np.asarray(
            meta["init_robot_base_ori"], dtype=np.float64
        )
        arrays["anchor_base_xyz_world"] = np.asarray(anchor_xyz, dtype=np.float64)
        arrays["anchor_base_rpy_world"] = np.asarray(anchor_rpy, dtype=np.float64)
        metadata = {
            "task": a.task,
            "episode": a.episode,
            "frame": 0,
            "scene_id": f"layout{meta.get('layout_id')}_style{meta.get('style_id')}",
            "anchor_source": "RoboCasa task reset after restoring episode fixture XML",
            "anchor_review_status": "unreviewed_provisional",
            "anchor_pose_reconstruction_check": anchor_pose_check,
            "candidate_consistency": candidate_consistency,
            "matched_xml_fixtures": matched_fixtures,
            "robot_geom_count": int(len(geom_ids)),
            "camera_coordinate_convention": "T_world_opencv_camera; x right, y down, z forward",
            "human_state_pose_semantics": "LeRobot observation.state[0:3] is the robot base center-site world position; quaternion [3:7] is XYZW",
            "metadata_init_pose_semantics": "ep_meta.init_robot_base_pos/orientation is the robot base-root reset pose",
            "human_state_site_height_minus_root_m": float(observation_state0[2] - float(meta["init_robot_base_pos"][2])),
            "resolution_hw": [h, w],
            "cameras": cam_meta,
        }
        arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
        a.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.output, **arrays)
        print(json.dumps(metadata, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
