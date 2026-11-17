"""Small CPU checks: accumulation preserves all three losses and one update."""
from types import SimpleNamespace

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from openpi.models.pi0_dual_teacher import masked_smooth_l1
from openpi.training.utils import TrainState
from scripts.train import train_step


class ToyDual(nnx.Module):
    def __init__(self):
        self.layer = nnx.Linear(3, 8, rngs=nnx.Rngs(1))

    def prepare_dual_inputs(self, rng, ob, action, scene, smask, ee, emask, *, train):
        # Full-batch randomness must stay fixed as the microbatch count changes.
        noise = jax.random.normal(rng, action.shape) * .1
        return ob, action + noise, action, jnp.zeros(len(ob)), scene, smask, ee, emask

    def compute_prepared_dual_losses(self, ob, x, target, time, scene, smask, ee, emask):
        prediction = self.layer(ob)
        a = jnp.mean((prediction[:, :2] - x) ** 2)
        sc, scn = masked_smooth_l1(prediction[:, 2:6], scene, smask)
        el, en = masked_smooth_l1(prediction[:, 6:], ee, emask)
        return a, sc, el, scn, en, prediction[:, 2:6], prediction[:, 6:]

    def compute_dual_losses(self, rng, *batch, train):
        return self.compute_prepared_dual_losses(*self.prepare_dual_inputs(rng, *batch, train=train))


@pytest.mark.parametrize("empty", [False, True])
def test_dual_accumulation_preserves_global_masked_means(empty):
    model = ToyDual()
    graph, params = nnx.split(model)
    tx = optax.chain(optax.clip_by_global_norm(1.), optax.sgd(.03))
    state = TrainState(step=jnp.asarray(0), params=params, model_def=graph, tx=tx,
                       opt_state=tx.init(params), ema_decay=None, ema_params=None)
    ob = jnp.arange(96, dtype=jnp.float32).reshape(32, 3) / 100
    smask = jnp.arange(128).reshape(32, 4) > 110
    emask = jnp.arange(64).reshape(32, 2) % 3 == 0
    if empty:
        smask = jnp.zeros_like(smask)
        emask = jnp.zeros_like(emask)
    # Invalid NaN targets must not enter the auxiliary losses or gradients.
    batch = (ob, jnp.ones((32, 2)), jnp.where(smask, .5, jnp.nan), smask,
             jnp.where(emask, -.2, jnp.nan), emask)
    outputs = []
    for count in [1, 8, 16]:
        config = SimpleNamespace(gradient_accumulation_steps=count, batch_size=32,
                                 trainable_filter=nnx.Param, lambda_scene=.1, lambda_ee=.1)
        output = jax.jit(lambda s: train_step(config, jax.random.key(2), s, batch))(state)
        jax.block_until_ready(output)
        outputs.append(output)
    for updated, info in outputs[1:]:
        assert int(updated.step) == 1
        for key in ["loss", "action_loss", "scene_loss", "ee_loss", "grad_norm", "scene_valid_count", "ee_valid_count"]:
            np.testing.assert_allclose(info[key], outputs[0][1][key], rtol=2e-6, atol=1e-6)
        for actual, expected in zip(jax.tree.leaves(updated.params), jax.tree.leaves(outputs[0][0].params)):
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)
