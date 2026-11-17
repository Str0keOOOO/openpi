"""One real-data/checkpoint smoke check for the spatial π0.5 model.

This is a preflight, not training: it performs one action-loss forward/backward,
one synthetic-target spatial-loss plumbing check, and one sampler call. The
synthetic target is explicitly labeled in the result and is never saved as a
teacher cache.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

from openpi.shared import nnx_utils
from openpi.training import config as config_lib, data_loader
from examples.robocasa_spatial.validation_data import single_window_csv


def _grad_norm(model, loss_fn, pattern):
    diff = nnx.DiffState(0, nnx_utils.PathRegex(pattern))
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff)(model)
    return float(loss), float(optax.global_norm(grads))


def main():
    config_name = os.environ.get("SPATIAL_CONFIG", "pi05_robocasa_spatial_query_human4")
    output = Path(os.environ.get(
        "SPATIAL_SMOKE_OUTPUT",
        "/data4/xuchenfei/robocasa365/output/anchor_visual_audit/real_model_validation.json",
    ))
    cfg = config_lib.get_config(config_name)
    teacher_root = os.environ.get("SPATIAL_TEACHER_CACHE_DIR")
    data_overrides = {"minimum_teacher_episode_coverage": 0.0}
    if teacher_root:
        # Diagnostic-only caches (for example, a one-episode provisional
        # SHARP cache) can be checked without weakening the formal config.
        data_overrides.update(spatial_teacher_cache_dir=teacher_root, spatial_split="all")
    # Missing teacher labels are allowed only in this diagnostic loader. They
    # remain fully masked, while the formal training config retains its 95% gate.
    sample_csv = None
    sample_provenance = None
    sample_task = os.environ.get("SPATIAL_SAMPLE_TASK")
    sample_episode = os.environ.get("SPATIAL_SAMPLE_EPISODE")
    sample_frame = os.environ.get("SPATIAL_SAMPLE_FRAME")
    if sample_task is not None or sample_episode is not None:
        if sample_task is None or sample_episode is None:
            raise ValueError("set both SPATIAL_SAMPLE_TASK and SPATIAL_SAMPLE_EPISODE")
        sample_csv, selected_frame = single_window_csv(
            cfg.data.spatial_windows_csv, sample_task, int(sample_episode),
            None if sample_frame is None else int(sample_frame),
        )
        sample_provenance = {"task": sample_task, "episode_index": int(sample_episode), "frame_index": selected_frame}
        data_overrides.update(spatial_windows_csv=str(sample_csv), spatial_split="all")
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
        observation, actions, teacher, teacher_valid = next(iter(loader))
    finally:
        if sample_csv is not None:
            sample_csv.unlink(missing_ok=True)
    if actions.shape != (1, cfg.model.action_horizon, cfg.model.action_dim):
        raise RuntimeError(f"unexpected action shape {actions.shape}")
    if teacher.shape != (1, 64) or teacher_valid.shape != teacher.shape:
        raise RuntimeError(f"unexpected teacher contract: {teacher.shape}, {teacher_valid.shape}")

    rng = jax.random.key(cfg.seed)
    model0 = cfg.model.create(rng)
    _, initial = nnx.split(model0)
    params = cfg.weight_loader.load(initial.to_pure_dict())
    model = cfg.model.load(params)
    model.eval()
    forward_rng = jax.random.fold_in(rng, 71)

    def action_fn(m):
        return m.compute_spatial_losses(
            forward_rng, observation, actions, teacher, teacher_valid, train=False
        )[0]

    action_loss, action_query_grad = _grad_norm(model, action_fn, ".*spatial_query.*")
    _, action_lora_grad = _grad_norm(model, action_fn, ".*lora.*")
    action_details = model.compute_spatial_losses(
        forward_rng, observation, actions, teacher, teacher_valid,
        train=False, return_action_prediction=True,
    )
    action_prediction = action_details[-1]

    # Isolate graph connectivity for the new auxiliary loss with an explicitly
    # synthetic target. This does not claim SHARP supervision is available.
    dummy_teacher = jnp.zeros((1, 64), dtype=jnp.float32)
    dummy_valid = jnp.ones((1, 64), dtype=jnp.bool_)

    def spatial_fn(m):
        return m.compute_spatial_losses(
            forward_rng, observation, actions, dummy_teacher, dummy_valid, train=False
        )[1]

    spatial_loss, spatial_query_grad = _grad_norm(model, spatial_fn, ".*spatial_query.*")
    _, spatial_head_grad = _grad_norm(model, spatial_fn, ".*geometry_head.*")

    real_teacher_cells = int(jnp.sum(teacher_valid))
    real_spatial_loss = None
    real_spatial_query_grad = None
    real_spatial_head_grad = None
    diagnostic_lambda = None
    if real_teacher_cells:
        def real_spatial_fn(m):
            return m.compute_spatial_losses(
                forward_rng, observation, actions, teacher, teacher_valid, train=False
            )[1]

        real_spatial_loss, real_spatial_query_grad = _grad_norm(
            model, real_spatial_fn, ".*spatial_query.*"
        )
        _, real_spatial_head_grad = _grad_norm(model, real_spatial_fn, ".*geometry_head.*")
        diagnostic_lambda = float(jnp.clip(0.1 * action_query_grad / real_spatial_query_grad, 1e-4, 1.0))

    sample = model.sample_actions(jax.random.fold_in(rng, 72), observation)
    if sample.shape != (1, cfg.model.action_horizon, cfg.model.action_dim):
        raise RuntimeError(f"unexpected sample_actions shape {sample.shape}")
    finite = bool(jnp.isfinite(sample).all())
    if not finite:
        raise RuntimeError("sample_actions returned non-finite values")

    result = {
        "status": "real_checkpoint_data_forward_backward_sampler_passed",
        "config": config_name,
        "device": str(jax.devices()[0]),
        "checkpoint": cfg.weight_loader.params_path,
        "action_shape": list(actions.shape),
        "teacher_shape": list(teacher.shape),
        "real_teacher_valid_cells": real_teacher_cells,
        "teacher_cache_root": teacher_root or cfg.data.spatial_teacher_cache_dir,
        "sample_provenance": sample_provenance,
        "normalized_action_target_abs_max": float(jnp.max(jnp.abs(actions))),
        "normalized_action_prediction_abs_max": float(jnp.max(jnp.abs(action_prediction))),
        "action_loss": action_loss,
        "action_loss_spatial_query_grad_norm": action_query_grad,
        "action_loss_action_expert_lora_grad_norm": action_lora_grad,
        "synthetic_spatial_loss": spatial_loss,
        "synthetic_spatial_loss_query_grad_norm": spatial_query_grad,
        "synthetic_spatial_loss_geometry_head_grad_norm": spatial_head_grad,
        "synthetic_target_note": "Only a plumbing check; not a real SHARP label and not used for training or cache.",
        "real_teacher_spatial_loss": real_spatial_loss,
        "real_teacher_spatial_query_grad_norm": real_spatial_query_grad,
        "real_teacher_spatial_head_grad_norm": real_spatial_head_grad,
        "provisional_lambda_spatial_diagnostic": diagnostic_lambda,
        "provisional_lambda_formula": "clip(0.1 * action_loss_query_gradient_norm / real_teacher_spatial_loss_query_gradient_norm, 1e-4, 1.0)",
        "real_teacher_note": (
            "Real SHARP teacher data from the configured cache; provisional status is retained in source metadata."
            if real_teacher_cells else "No valid real teacher cells were present in this sample."
        ),
        "sample_actions_shape": list(sample.shape),
        "sample_actions_finite": finite,
    }
    for k in (
        "action_loss_spatial_query_grad_norm",
        "action_loss_action_expert_lora_grad_norm",
        "synthetic_spatial_loss_query_grad_norm",
        "synthetic_spatial_loss_geometry_head_grad_norm",
    ):
        if not jnp.isfinite(result[k]) or result[k] <= 0:
            result["status"] = "failed_gradient_check"
            raise RuntimeError(f"{k} expected finite nonzero value; got {result[k]}")
    if real_teacher_cells and (
        not jnp.isfinite(real_spatial_loss)
        or not jnp.isfinite(real_spatial_query_grad)
        or not jnp.isfinite(real_spatial_head_grad)
        or real_spatial_query_grad <= 0
        or real_spatial_head_grad <= 0
    ):
        raise RuntimeError("real SHARP teacher loss must give finite nonzero Query and GeometryHead gradients")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
