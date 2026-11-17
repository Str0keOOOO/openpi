"""Spatial-query extension of π0 with a masked parallax-latent objective.

The registered RoboCasa config appends learnable query tokens to the VLM prefix.
The action suffix attends to those tokens; a GeometryHead predicts the
frame-aligned teacher latent during training. Inference requires no teacher.
"""
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from openpi.models import gemma
from openpi.models import model as model_lib
from openpi.models.pi0 import Pi0, make_attn_mask


class Pi0Spatial(Pi0):
    def __init__(self, config, rngs, *, num_queries=8, teacher_dim=64):
        if not config.pi05:
            raise ValueError('Spatial queries currently require pi05 discrete-state prefix')
        if num_queries < 1 or teacher_dim != 64:
            raise ValueError('Expected positive query count and 64 teacher dimensions (two 32D agentview latents)')
        super().__init__(config, rngs)
        self.num_queries = num_queries
        width = gemma.get_config(config.paligemma_variant).width
        self.spatial_query = nnx.Param(jax.random.normal(rngs.params(), (num_queries, width)) * 0.02)
        self.geometry_head = nnx.Linear(width, teacher_dim, rngs=rngs)

    def embed_prefix(self, observation):
        tokens, mask, ar = super().embed_prefix(observation)
        q = jnp.broadcast_to(self.spatial_query.value[None], (tokens.shape[0], self.num_queries, tokens.shape[-1]))
        return (jnp.concatenate([tokens, q], axis=1),
                jnp.concatenate([mask, jnp.ones((mask.shape[0], self.num_queries), dtype=bool)], axis=1),
                jnp.concatenate([ar, jnp.zeros((self.num_queries,), dtype=bool)], axis=0))

    def compute_spatial_losses(
        self, rng, observation, actions, teacher, teacher_valid, *,
        train=False, return_action_prediction=False,
    ):
        """Return action loss, masked spatial loss, and per-camera spatial losses.

        ``teacher_valid`` is Bx64; missing samples must provide all-false masks.
        The Action Expert shares the prefix attention context with SpatialQuery,
        so the action loss can train queries even when the auxiliary label is
        absent.
        """
        if teacher.shape[-1] != 64 or teacher_valid.shape != teacher.shape:
            raise ValueError('teacher and teacher_valid must have matching Bx64 shapes')
        prep_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = model_lib.preprocess_observation(prep_rng, observation, train=train)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        x = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
        target = noise - actions
        prefix, pmask, par = self.embed_prefix(observation)
        suffix, smask, sar, cond = self.embed_suffix(observation, x, time)
        mask = jnp.concatenate([pmask, smask], axis=1)
        attention = make_attn_mask(mask, jnp.concatenate([par, sar], axis=0))
        positions = jnp.cumsum(mask, axis=1) - 1
        (pout, sout), _ = self.PaliGemma.llm([prefix, suffix], mask=attention, positions=positions, adarms_cond=[None, cond])
        action_prediction = self.action_out_proj(sout[:, -self.action_horizon:])
        action_error=jnp.square(action_prediction-target)
        action_loss=jnp.mean(action_error)
        predicted = self.geometry_head(jnp.mean(pout[:, -self.num_queries:], axis=1))
        valid = teacher_valid.astype(bool)
        safe_target = jnp.where(valid, teacher, 0.0)
        diff = jnp.where(valid, predicted-safe_target, 0.0)
        abs_diff = jnp.abs(diff)
        element = jnp.where(abs_diff < 1, .5 * diff**2, abs_diff - .5)
        total_count = jnp.sum(valid)
        spatial_loss = jnp.sum(element) / jnp.maximum(total_count, 1)
        per_camera = jnp.sum(element.reshape((-1, 2, 32)), axis=(0, 2)) / jnp.maximum(jnp.sum(valid.reshape((-1, 2, 32)), axis=(0, 2)), 1)
        result = (action_loss, spatial_loss, per_camera, predicted, total_count)
        if return_action_prediction:
            return (*result, action_prediction)
        return result
