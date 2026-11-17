"""Generate real SHARP episode-level 64D parallax caches from staged geometry.

This entry point runs inside the independent SHARP environment. It launches the
RoboCasa Python environment only for scene calibration/segmentation export, so
PyTorch/gsplat and MuJoCo dependencies remain isolated from each other.
The retained training episode list determines which Anchor geometries are used.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch

from examples.robocasa_spatial.geometry import masked_average_pool, parallax
from examples.robocasa_spatial.teacher_cache import CAMERAS, save


PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path("/data4/xuchenfei/robocasa365")
DATA_ROOT = ROOT / "third_party/robocasa/datasets/v1.0/pretrain/atomic"
AUDIT_ROOT = ROOT / "output/anchor_visual_audit"
DEFAULT_CHECKPOINT = ROOT / "model/sharp/sharp_2572gikvuh.pt"
DEFAULT_GEOMETRY = AUDIT_ROOT / "geometry_staging"
DEFAULT_EXCLUSIONS = AUDIT_ROOT / "manual_episode_exclusions.csv"
TASKS = ("PickPlaceSinkToCounter", "OpenDrawer", "OpenStandMixerHead", "PickPlaceCounterToCabinet")
VIDEO_KEYS = {
    "robot0_agentview_left": "observation.images.robot0_agentview_left",
    "robot0_agentview_right": "observation.images.robot0_agentview_right",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare_gsplat_toolchain() -> None:
    """Make the SHARP env's Ninja/CUDA tools visible to gsplat's JIT build."""
    try:
        import ninja
    except ImportError as exc:
        raise RuntimeError("gsplat JIT rendering needs Ninja in the SHARP environment") from exc
    ninja_bin = Path(ninja.BIN_DIR)
    os.environ["PATH"] = str(ninja_bin) + os.pathsep + os.environ.get("PATH", "")
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    nvcc = cuda_home / "bin" / "nvcc"
    if nvcc.is_file():
        os.environ.setdefault("CUDA_HOME", str(cuda_home))
        os.environ["PATH"] = str(cuda_home / "bin") + os.pathsep + os.environ["PATH"]


def read_video_frame0(path: Path) -> np.ndarray:
    gen = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        meta = next(gen)
        width, height = meta["size"]
        frame = np.frombuffer(next(gen), dtype=np.uint8).reshape(height, width, 3).copy()
    except Exception as exc:
        raise RuntimeError(f"cannot decode RGB frame 0 from {path}: {exc}") from exc
    finally:
        gen.close()
    return frame


def load_eligible(anchor_csv: Path):
    with (ROOT / "data/training_views/episodes.csv").open(newline="") as f:
        retained = {(r["task"], int(r["episode_index"])) for r in csv.DictReader(f)}
    with anchor_csv.open(newline="") as f:
        candidates = {(r["task"], int(r["episode_index"])): r for r in csv.DictReader(f)}
    missing = retained - candidates.keys()
    if missing:
        raise ValueError(f"Missing Anchor geometry for retained episodes: {sorted(missing)}")
    return {key: candidates[key] for key in sorted(retained)}

def stage_geometry(task: str, episode: int, path: Path, python: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [python, "-m", "examples.robocasa_spatial.export_episode_geometry",
           "--task", task, "--episode", str(episode), "--output", str(path)]
    subprocess.run(cmd, cwd=PROJECT, env=env, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--anchor-csv", type=Path, default=AUDIT_ROOT / "anchor_candidates.csv")
    p.add_argument("--sharp-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--output", type=Path)
    p.add_argument("--geometry-dir", type=Path, default=DEFAULT_GEOMETRY)
    p.add_argument("--robocasa-python", default="/data4/xuchenfei/robocasa365/envs/robocasa/bin/python")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    out = a.output or AUDIT_ROOT / "teacher_cache/provisional"
    if not a.sharp_checkpoint.is_file():
        p.error(f"SHARP checkpoint missing: {a.sharp_checkpoint}")
    if not a.limit and a.anchor_csv.resolve() == (AUDIT_ROOT / "anchor_candidates.csv").resolve():
        with a.anchor_csv.open(newline="") as f:
            audit_rows = list(csv.DictReader(f))
        counts = Counter(r["task"] for r in audit_rows)
        progress_file = AUDIT_ROOT / "progress.json"
        audit_progress = json.loads(progress_file.read_text()) if progress_file.is_file() else {}
        keys = {(r["task"], int(r["episode_index"])) for r in audit_rows}
        if len(keys) != 400 or any(counts[t] != 100 for t in TASKS) or audit_progress.get("processed") != 400:
            raise RuntimeError(
                "Full-cache generation requires a completed 400-episode Anchor audit "
                f"(unique={len(keys)}, per_task={dict(counts)}, progress={audit_progress.get('processed')}/400)."
            )
    chosen = load_eligible(a.anchor_csv)
    if a.limit:
        chosen = dict(list(sorted(chosen.items()))[:a.limit])
    if not chosen:
        raise RuntimeError("no retained episodes with Anchor geometry")
    if not torch.cuda.is_available():
        raise RuntimeError("SHARP generation requires CUDA + gsplat; no CUDA device is visible")

    # gsplat compiles its CUDA extension on first render. Running the env's
    # Python directly does not necessarily put its Ninja executable on PATH.
    prepare_gsplat_toolchain()

    from sharp.cli.predict import predict_image
    from sharp.models import PredictorParams, create_predictor
    from sharp.utils.gsplat import GSplatRenderer

    device = torch.device("cuda")
    state = torch.load(a.sharp_checkpoint, map_location="cpu", weights_only=True)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state)
    predictor.eval().to(device)
    renderer = GSplatRenderer()
    checkpoint_hash = sha256(a.sharp_checkpoint)
    a.geometry_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    failures = out / "failures.jsonl"
    progress_path = out / "progress.json"
    progress = {"selected": len(chosen),
                "completed": 0, "skipped_cache": 0, "failed": 0}

    for (task, episode), anchor_row in sorted(chosen.items()):
        cache = out / task / f"episode_{episode:06d}.npz"
        if cache.exists():
            try:
                with np.load(cache, allow_pickle=False) as z:
                    old = json.loads(str(z["metadata"]))
                if old.get("cache_version") == "episode-agentview-parallax-v1":
                    progress["skipped_cache"] += 1
                    continue
            except Exception:
                pass
        geometry_path = a.geometry_dir / f"{task}_ep{episode:06d}.npz"
        env_sim = None
        try:
            if not geometry_path.is_file():
                stage_geometry(task, episode, geometry_path, a.robocasa_python)
            with np.load(geometry_path, allow_pickle=False) as geom:
                geom = {k: geom[k].copy() for k in geom.files}
            geom_meta = json.loads(str(geom["metadata"]))
            if geom_meta.get("task") != task or int(geom_meta.get("episode", -1)) != episode:
                raise RuntimeError(f"staged geometry provenance mismatch: {geometry_path}")
            source_xyz = geom["human_base_xyz_world"].tolist()
            source_quat = geom["human_base_quaternion_xyzw"].tolist()
            anchor_xyz = geom["anchor_base_xyz_world"].tolist()
            anchor_rpy = geom["anchor_base_rpy_world"].tolist()
            root = next((DATA_ROOT / task).glob("*/lerobot"))
            latents, masks = {}, {}
            robot_source, robot_anchor, flow_masks, flows = {}, {}, {}, {}
            depths, alphas = {}, {}
            cam_meta = {}

            for camera, video_key in VIDEO_KEYS.items():
                video = root / "videos" / f"chunk-{episode//1000:03d}" / video_key / f"episode_{episode:06d}.mp4"
                rgb = read_video_frame0(video)
                h, w = rgb.shape[:2]
                k = geom[f"K_{camera}"].astype(np.float64)
                t_world_source = geom[f"T_world_source_{camera}"].astype(np.float64)
                t_world_anchor = geom[f"T_world_anchor_{camera}"].astype(np.float64)
                robot_src = geom[f"robot_mask_source_{camera}"].astype(bool)
                robot_ref = geom[f"robot_mask_anchor_{camera}"].astype(bool)
                if rgb.shape[:2] != robot_src.shape or robot_src.shape != robot_ref.shape:
                    raise RuntimeError(f"RGB and segmentation mask shapes do not match for {camera}")
                f_px = float(k[0, 0])
                gaussians = predict_image(predictor, rgb, f_px, device)
                k_t = torch.as_tensor(k, dtype=torch.float32, device=device)[None]
                identity = torch.eye(4, dtype=torch.float32, device=device)[None]
                reference_from_source = np.linalg.inv(t_world_anchor) @ t_world_source
                source_from_reference = np.linalg.inv(reference_from_source)
                ref_view = torch.as_tensor(reference_from_source, dtype=torch.float32, device=device)[None]
                with torch.no_grad():
                    cur = renderer(gaussians, identity, k_t, w, h)
                    ref = renderer(gaussians, ref_view, k_t, w, h)
                dcur = cur.depth[0, 0].float().cpu().numpy()
                acur = cur.alpha[0, 0].float().cpu().numpy()
                dref = ref.depth[0, 0].float().cpu().numpy()
                aref = ref.alpha[0, 0].float().cpu().numpy()
                valid_cur = np.isfinite(dcur) & (dcur > 0) & (acur > 0.5) & (~robot_src)
                valid_ref = np.isfinite(dref) & (dref > 0) & (aref > 0.5) & (~robot_ref)
                field, valid = parallax(
                    dcur, k, np.eye(4), source_from_reference, valid_cur,
                    reference_depth=dref, reference_valid=valid_ref,
                )
                latent, latent_mask = masked_average_pool(field, valid, size=4)
                latents[camera] = latent
                masks[camera] = latent_mask
                robot_source[camera] = robot_src
                robot_anchor[camera] = robot_ref
                flow_masks[camera] = valid
                flows[camera] = field
                depths[camera] = np.stack([dcur, dref], axis=0).astype(np.float16)
                alphas[camera] = np.stack([acur, aref], axis=0).astype(np.float16)
                cam_meta[camera] = {
                    "intrinsics": k.tolist(),
                    "world_from_current_opencv": t_world_source.tolist(),
                    "world_from_anchor_opencv": t_world_anchor.tolist(),
                    "focal_px": f_px,
                    "robot_mask_fraction_current": float(robot_src.mean()),
                    "robot_mask_fraction_anchor": float(robot_ref.mean()),
                    "valid_flow_fraction": float(valid.mean()),
                    "sharp_depth_median_current_m": float(np.median(dcur[valid_cur])) if valid_cur.any() else None,
                    "sharp_depth_median_anchor_m": float(np.median(dref[valid_ref])) if valid_ref.any() else None,
                    "latent_valid_cells": int(latent_mask.sum()),
                }
                del gaussians, cur, ref
                torch.cuda.empty_cache()

            metadata = {
                "task": task, "episode": episode, "frame": 0,
                "sharp_version": "apple-ml-sharp-0.1-local", "sharp_checkpoint_sha256": checkpoint_hash,
                "anchor_source": geom_meta["anchor_source"],
                "candidate_consistency": geom_meta.get("candidate_consistency"),
                "calibration_id": "robocasa_xml_mujoco_camera_segmentation_v1",
                "cache_version": "episode-agentview-parallax-v1",
                "teacher_scope": "episode_initial_static_scene",
                "human_base_xyz_world": source_xyz,
                "human_base_quaternion_xyzw": source_quat,
                "anchor_base_xyz_world": anchor_xyz,
                "anchor_base_rpy_world_rad": anchor_rpy,
                "camera_coordinate_convention": geom_meta["camera_coordinate_convention"],
                "parallax_direction": "F(u,v)=(u_anchor-u_current,v_anchor-v_current)",
                "flow_normalization": "du/image_width,dv/image_height",
                "pooling": "masked adaptive average 4x4 per two agentviews; concatenate left then right",
                "rgb_source": "original RoboCasa Human video frame 0, not the simulation review render",
                "cameras": cam_meta,
            }
            # Preserve intermediate geometry for audit/recalculation, but never
            # retain the much larger per-episode 3D Gaussian PLY.
            save(cache, latents, masks, metadata,
                 pixel_robot_masks_current=robot_source,
                 pixel_robot_masks_reference=robot_anchor,
                 pixel_flow_masks=flow_masks)
            tmp = cache.with_suffix(".diag.tmp.npz")
            np.savez_compressed(tmp, **{f"flow_{k}": v for k, v in flows.items()},
                                **{f"depth_{k}": v for k, v in depths.items()},
                                **{f"alpha_{k}": v for k, v in alphas.items()})
            tmp.replace(cache.with_name(cache.stem + "_diagnostics.npz"))
            progress["completed"] += 1
        except Exception as exc:
            progress["failed"] += 1
            with failures.open("a") as f:
                f.write(json.dumps({"task": task, "episode": episode, "error": f"{type(exc).__name__}: {exc}",
                                    "traceback": traceback.format_exc()}, ensure_ascii=False) + "\n")
        progress["last"] = {"task": task, "episode": episode}
        progress_path.write_text(json.dumps(progress, indent=2))
        print(json.dumps({**progress, "last": progress["last"]}), flush=True)
    progress_path.write_text(json.dumps(progress, indent=2))
    print(json.dumps(progress, indent=2))


if __name__ == "__main__":
    main()
