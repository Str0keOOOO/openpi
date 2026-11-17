"""Run one isolated optimizer update and native checkpoint round-trip.

This is a one-batch engineering validation, not a training run. It requires a
real SHARP cache and keeps its review status intact in the output report.
"""
from __future__ import annotations

import dataclasses
import functools
import json
import os
import shutil
import tempfile
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.shared import nnx_utils
from openpi.training import checkpoints, config as config_lib, data_loader, optimizer, utils
from scripts.train import train_step
from examples.robocasa_spatial.validation_data import single_window_csv


def _pure(state):
    return state.to_pure_dict()


def _group_delta(before, after, marker):
    b, a = _pure(before), _pure(after)
    total = jnp.asarray(0.0, jnp.float32)
    count = 0
    for path, x in jax.tree_util.tree_flatten_with_path(b)[0]:
        key = jax.tree_util.keystr(path)
        if marker in key:
            y = a
            for part in path:
                y = y[part.key if hasattr(part, "key") else part.idx if hasattr(part, "idx") else part.name]
            total = total + jnp.sum(jnp.square(jnp.asarray(y, jnp.float32) - jnp.asarray(x, jnp.float32)))
            count += int(np.prod(x.shape))
    return float(jnp.sqrt(total)), count


def _frozen_probe(before, after):
    b, a = _pure(before), _pure(after)
    flat = jax.tree_util.tree_flatten_with_path(b)[0]
    for path, x in flat:
        key = jax.tree_util.keystr(path)
        if any(s in key.lower() for s in ("lora", "spatial_query", "geometry_head")):
            continue
        y = a
        try:
            for part in path:
                y = y[part.key if hasattr(part, "key") else part.idx if hasattr(part, "idx") else part.name]
        except (KeyError, IndexError, TypeError):
            continue
        if getattr(x, "size", 0) and getattr(y, "size", 0):
            x_probe = jnp.asarray(x).reshape(-1)[:64]
            y_probe = jnp.asarray(y).reshape(-1)[:64]
            return key, float(jnp.max(jnp.abs(y_probe.astype(jnp.float32)-x_probe.astype(jnp.float32))))
    raise RuntimeError("could not find a frozen parameter probe")


def main():
    config_name = os.environ.get("SPATIAL_CONFIG", "pi05_robocasa_spatial_query_human4")
    teacher_root = os.environ.get("SPATIAL_TEACHER_CACHE_DIR")
    if not teacher_root:
        raise RuntimeError("Set SPATIAL_TEACHER_CACHE_DIR to a real SHARP cache (provisional is allowed for diagnostics).")
    output = Path(os.environ.get(
        "SPATIAL_OPTIMIZER_VALIDATION_OUTPUT",
        "/data4/xuchenfei/robocasa365/output/anchor_visual_audit/optimizer_checkpoint_validation.json",
    ))
    cfg = config_lib.get_config(config_name)
    sample_csv = None
    sample_provenance = None
    sample_task = os.environ.get("SPATIAL_SAMPLE_TASK")
    sample_episode = os.environ.get("SPATIAL_SAMPLE_EPISODE")
    sample_frame = os.environ.get("SPATIAL_SAMPLE_FRAME")
    data_overrides = dict(
        spatial_teacher_cache_dir=teacher_root,
        spatial_split="all",
        minimum_teacher_episode_coverage=0.0,
    )
    if sample_task is not None or sample_episode is not None:
        if sample_task is None or sample_episode is None:
            raise ValueError("set both SPATIAL_SAMPLE_TASK and SPATIAL_SAMPLE_EPISODE")
        sample_csv, selected_frame = single_window_csv(
            cfg.data.spatial_windows_csv, sample_task, int(sample_episode),
            None if sample_frame is None else int(sample_frame),
        )
        data_overrides["spatial_windows_csv"] = str(sample_csv)
        sample_provenance = {"task": sample_task, "episode_index": int(sample_episode), "frame_index": selected_frame}
    cfg = dataclasses.replace(
        cfg,
        data=dataclasses.replace(cfg.data, **data_overrides),
        batch_size=1,
        num_workers=0,
    )
    try:
        loader = data_loader.create_data_loader(
            cfg, shuffle=False, num_batches=1, skip_norm_stats=False, framework="jax"
        )
        batch = next(iter(loader))
    finally:
        if sample_csv is not None:
            sample_csv.unlink(missing_ok=True)
    observation, actions, teacher, teacher_valid, action_mask = batch
    valid_cells = int(jnp.sum(teacher_valid))
    if valid_cells <= 0:
        raise RuntimeError("first real Human window has no valid SHARP teacher cells")
    if actions.shape != (1, cfg.model.action_horizon, cfg.model.action_dim):
        raise RuntimeError(f"action batch has unexpected shape {actions.shape}")

    rng = jax.random.key(cfg.seed)
    model0 = cfg.model.create(rng)
    _, initial = nnx.split(model0)
    loaded = cfg.weight_loader.load(initial.to_pure_dict())
    model = cfg.model.load(loaded)
    params = nnx_utils.state_map(
        nnx.state(model), cfg.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16))
    )
    tx = optimizer.create_optimizer(cfg.optimizer, cfg.lr_schedule, weight_decay_mask=None)
    state = utils.TrainState(
        step=jnp.asarray(0, jnp.int32),
        params=params,
        model_def=nnx.graphdef(model),
        tx=tx,
        opt_state=tx.init(params.filter(cfg.trainable_filter)),
        ema_decay=None,
        ema_params=None,
    )
    rng = jax.random.fold_in(rng, 731)
    step_fn = jax.jit(functools.partial(train_step, cfg))
    updated, metrics = step_fn(rng, state, batch)
    jax.block_until_ready(metrics)
    values = {k: float(jax.device_get(v)) for k, v in metrics.items()}
    if not all(np.isfinite(v) for v in values.values()):
        raise RuntimeError(f"one-step train metrics are non-finite: {values}")
    if int(jax.device_get(updated.step)) != 1:
        raise RuntimeError("optimizer update did not advance exactly one diagnostic step")

    deltas = {}
    delta_counts = {}
    for marker in ("spatial_query", "geometry_head", "lora"):
        deltas[marker], counts = _group_delta(state.params, updated.params, marker)
        delta_counts[marker] = counts
        if counts <= 0:
            raise RuntimeError(f"could not find trainable parameters for group {marker}")
        if deltas[marker] <= 0:
            raise RuntimeError(f"{marker} parameters did not change in the optimizer update")
    frozen_path, frozen_max_abs_delta = _frozen_probe(state.params, updated.params)
    if frozen_max_abs_delta != 0:
        raise RuntimeError(f"frozen parameter changed at {frozen_path}: {frozen_max_abs_delta}")

    # Use the repository's actual checkpoint manager and save/restore helpers.
    # The temporary checkpoint contains the full params and optimizer state;
    # remove it after successful round-trip to avoid retaining a duplicate base.
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_root = Path(tempfile.mkdtemp(prefix="spatial_roundtrip_", dir=output.parent))
    manager = None
    try:
        manager, _ = checkpoints.initialize_checkpoint_dir(
            checkpoint_root, keep_period=None, overwrite=True, resume=False
        )
        checkpoints.save_state(manager, updated, loader, int(jax.device_get(updated.step)))
        manager.wait_until_finished()
        restored = checkpoints.restore_state(
            manager, updated, loader, int(jax.device_get(updated.step))
        )
        if int(jax.device_get(restored.step)) != 1:
            raise RuntimeError("restored checkpoint has incorrect optimizer step")
        restored_model = nnx.merge(restored.model_def, restored.params)
        sample = restored_model.sample_actions(jax.random.fold_in(rng, 1), observation)
        if sample.shape != (1, cfg.model.action_horizon, cfg.model.action_dim):
            raise RuntimeError(f"restored model inference shape mismatch: {sample.shape}")
        if not bool(jnp.isfinite(sample).all()):
            raise RuntimeError("restored model inference returned non-finite actions")
        # Compare the serialized trainable state exactly, without copying the
        # full frozen 3B-parameter backbone to host memory.
        train_before = updated.params.filter(cfg.trainable_filter)
        train_after = restored.params.filter(cfg.trainable_filter)
        roundtrip_error = float(optax.global_norm(jax.tree.map(
            lambda x, y: jnp.asarray(x, jnp.float32)-jnp.asarray(y, jnp.float32),
            train_before, train_after,
        )))
        if roundtrip_error != 0:
            raise RuntimeError(f"trainable parameter checkpoint round-trip error is {roundtrip_error}")
        status = "one_real_batch_update_and_native_checkpoint_roundtrip_passed"
    finally:
        if manager is not None:
            manager.wait_until_finished()
            manager.close()
        shutil.rmtree(checkpoint_root, ignore_errors=True)

    result = {
        "status": status,
        "config": config_name,
        "device": str(jax.devices()[0]),
        "checkpoint": cfg.weight_loader.params_path,
        "teacher_cache_root": teacher_root,
        "sample_provenance": sample_provenance,
        "real_teacher_valid_cells": valid_cells,
        "action_shape": list(actions.shape),
        "one_step_metrics": values,
        "trainable_parameter_delta_l2": deltas,
        "trainable_parameter_counts": delta_counts,
        "frozen_parameter_probe_path": frozen_path,
        "frozen_parameter_probe_max_abs_delta": frozen_max_abs_delta,
        "checkpoint_trainable_params_roundtrip_l2_error": roundtrip_error,
        "restored_sampler_shape": list(sample.shape),
        "restored_sampler_finite": True,
        "formal_training_started": False,
        "temporary_full_checkpoint_removed": True,
        "teacher_status_note": "Uses real SHARP output, but its Anchor is unreviewed; diagnostic only, not formal calibration/training data.",
    }
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
