"""π0.5 spatial-query model with independent 64D scene and 6D EE heads."""
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from openpi.models import gemma
from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0_spatial import Pi0Spatial


def masked_smooth_l1(prediction, target, valid):
    if prediction.shape != target.shape or valid.shape != target.shape:
        raise ValueError('Prediction, target, and mask must have the same shape')
    valid = valid.astype(bool)
    difference = jnp.where(valid, prediction-jnp.where(valid, target, 0.), 0.)
    magnitude = jnp.abs(difference)
    element = jnp.where(magnitude < 1., .5*difference**2, magnitude-.5)
    count = jnp.sum(valid)
    return jnp.sum(element)/jnp.maximum(count, 1), count


class Pi0DualTeacher(Pi0Spatial):
    def __init__(self, config, rngs, *, num_queries=8, teacher_dim=64):
        super().__init__(config, rngs, num_queries=num_queries, teacher_dim=teacher_dim)
        del self.geometry_head
        width = gemma.get_config(config.paligemma_variant).width
        self.scene_hidden = nnx.Linear(width, 512, rngs=rngs)
        self.scene_head = nnx.Linear(512, 64, rngs=rngs)
        self.ee_hidden = nnx.Linear(width, 512, rngs=rngs)
        self.ee_head = nnx.Linear(512, 6, rngs=rngs)

    def prepare_dual_inputs(self, rng, observation, actions, scene, scene_valid, ee, ee_valid, *, train=False):
        """Prepare the whole effective batch before splitting it for accumulation.

        This keeps augmentation, flow-matching noise and valid-dimension counts
        identical to the non-accumulated objective for the same RNG and batch.
        No model parameters or activations are needed in this preparation.
        """
        dense_scene=scene.ndim==5
        if (not dense_scene and scene.shape[-1] != 64) or ee.shape[-1] != 6:
            raise ValueError('Expected scene Bx64 and EE Bx6')
        prep_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        # Dense caches follow the same native RGB augmentation, transforming
        # both flow endpoints before online pooling. Legacy pooled caches are
        # diagnostic-only and cannot support geometric augmentation.
        if dense_scene:
            from openpi.models import teacher_augmentation
            observation,scene,scene_valid=teacher_augmentation.preprocess(prep_rng,observation,scene,scene_valid,train=train)
        else:
            observation = model_lib.preprocess_observation(prep_rng, observation, train=False)
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2])*.999+.001
        x = time[...,None,None]*noise+(1.-time[...,None,None])*actions
        target = noise-actions
        return observation, x, target, time, scene, scene_valid, ee, ee_valid

    def compute_prepared_dual_losses(self, observation, x, target, time, scene, scene_valid, ee, ee_valid):
        """The model forward for an already augmented whole or micro batch."""
        prefix, pmask, par = self.embed_prefix(observation)
        suffix, smask, sar, cond = self.embed_suffix(observation,x,time)
        mask=jnp.concatenate([pmask,smask],axis=1)
        attention=make_attn_mask(mask,jnp.concatenate([par,sar],axis=0))
        positions=jnp.cumsum(mask,axis=1)-1
        (pout,sout),_=self.PaliGemma.llm([prefix,suffix],mask=attention,positions=positions,adarms_cond=[None,cond])
        action_prediction=self.action_out_proj(sout[:,-self.action_horizon:])
        action_loss=jnp.mean(jnp.square(action_prediction-target))
        query=jnp.mean(pout[:,-self.num_queries:],axis=1)
        scene_prediction=self.scene_head(jax.nn.gelu(self.scene_hidden(query)))
        ee_prediction=self.ee_head(jax.nn.gelu(self.ee_hidden(query)))
        scene_loss,scene_count=masked_smooth_l1(scene_prediction,scene,scene_valid)
        ee_loss,ee_count=masked_smooth_l1(ee_prediction,ee,ee_valid)
        return action_loss,scene_loss,ee_loss,scene_count,ee_count,scene_prediction,ee_prediction

    def compute_dual_losses(self, rng, observation, actions, scene, scene_valid, ee, ee_valid, *, train=False):
        prepared = self.prepare_dual_inputs(rng, observation, actions, scene, scene_valid, ee, ee_valid, train=train)
        return self.compute_prepared_dual_losses(*prepared)
