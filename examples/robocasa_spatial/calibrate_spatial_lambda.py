"""Calibrate spatial-loss weight from real teacher data and Query gradients.

An explicitly opted-in provisional cache produces a diagnostic recommendation
only; it must not be copied into the formal training config before Anchor review.
"""
import argparse
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


def grad_norm(model, loss_fn, path_regex):
    diff=nnx.DiffState(0,nnx_utils.PathRegex(path_regex))
    value,grads=nnx.value_and_grad(loss_fn,argnums=diff)(model)
    return float(value),float(optax.global_norm(grads))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='pi05_robocasa_spatial_query_human4')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--teacher-root',type=Path)
    p.add_argument('--allow-provisional',action='store_true')
    p.add_argument('--sample-task')
    p.add_argument('--sample-episode',type=int)
    p.add_argument('--sample-frame',type=int)
    a=p.parse_args();cfg=config_lib.get_config(a.config)
    if not getattr(cfg.model,'enable_spatial_query',False):raise ValueError('selected config does not enable SpatialQuery')
    teacher_root=a.teacher_root or Path(os.environ.get('SPATIAL_TEACHER_CACHE_DIR',cfg.data.spatial_teacher_cache_dir))
    formal_root=any(part=='formal' or part.endswith('_formal') for part in teacher_root.parts)
    provisional=not formal_root
    if provisional and not a.allow_provisional:
        raise ValueError('provisional teacher calibration requires --allow-provisional and remains diagnostic only')
    data_overrides={'spatial_teacher_cache_dir':str(teacher_root),'spatial_split':'all','minimum_teacher_episode_coverage':0.0}
    sample_csv=None;sample_provenance=None
    if a.sample_task is not None or a.sample_episode is not None:
        if a.sample_task is None or a.sample_episode is None:
            raise ValueError('--sample-task and --sample-episode must be given together')
        sample_csv,selected_frame=single_window_csv(cfg.data.spatial_windows_csv,a.sample_task,a.sample_episode,a.sample_frame)
        data_overrides['spatial_windows_csv']=str(sample_csv)
        sample_provenance={'task':a.sample_task,'episode_index':a.sample_episode,'frame_index':selected_frame}
    small=dataclasses.replace(cfg,data=dataclasses.replace(cfg.data,**data_overrides),batch_size=1,num_workers=0)
    try:
        loader=data_loader.create_data_loader(small,shuffle=False,num_batches=1,skip_norm_stats=False,framework='jax')
        batch=next(iter(loader))
    finally:
        if sample_csv is not None:sample_csv.unlink(missing_ok=True)
    if len(batch)!=4:raise RuntimeError('expected observation, actions, teacher, teacher-valid mask')
    observation,actions,teacher,valid=batch
    if not bool(jnp.any(valid)):
        raise RuntimeError('real calibration batch has no valid spatial teacher cells')

    key=jax.random.key(cfg.seed);model0=cfg.model.create(key)
    _,initial=nnx.split(model0)
    pure=initial.to_pure_dict()
    loaded=cfg.weight_loader.load(pure)
    model=cfg.model.load(loaded);model.eval()
    rng=jax.random.fold_in(key,101)
    def action_fn(m):
        return m.compute_spatial_losses(rng,observation,actions,teacher,valid,train=False)[0]
    def spatial_fn(m):
        return m.compute_spatial_losses(rng,observation,actions,teacher,valid,train=False)[1]
    action_loss,action_query_grad=grad_norm(model,action_fn,'.*spatial_query.*')
    spatial_loss,spatial_query_grad=grad_norm(model,spatial_fn,'.*spatial_query.*')
    _,spatial_head_grad=grad_norm(model,spatial_fn,'.*geometry_head.*')
    if not all(map(lambda x:jnp.isfinite(x),[action_loss,action_query_grad,spatial_loss,spatial_query_grad,spatial_head_grad])):
        raise RuntimeError('non-finite loss or gradient norm')
    if action_query_grad<=0 or spatial_query_grad<=0 or spatial_head_grad<=0:
        raise RuntimeError(f'expected nonzero shared gradients; got action->query={action_query_grad}, spatial->query={spatial_query_grad}, spatial->head={spatial_head_grad}')
    raw=0.1*action_query_grad/spatial_query_grad
    recommended=float(jnp.clip(raw,1e-4,1.0))
    result={'status':'provisional_diagnostic_only' if provisional else 'calibrated_from_real_batch','config':cfg.name,'batch_size':1,'valid_teacher_cells':int(jnp.sum(valid)),
        'action_loss':action_loss,'spatial_loss':spatial_loss,'action_loss_query_gradient_norm':action_query_grad,
        'spatial_loss_query_gradient_norm':spatial_query_grad,'spatial_loss_geometry_head_gradient_norm':spatial_head_grad,
        'target_aux_to_action_query_gradient_ratio':0.1,'unclipped_lambda':raw,'recommended_lambda_spatial':recommended,
        'formula':'clip(0.1 * ||grad_query L_action|| / ||grad_query L_spatial||, 1e-4, 1.0)',
        'checkpoint':cfg.weight_loader.params_path,'teacher_cache':str(teacher_root),
        'sample_provenance':sample_provenance,
        'formal_config_modified':False,
        'anchor_review_status':'unreviewed_provisional' if provisional else 'accepted_formal'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
