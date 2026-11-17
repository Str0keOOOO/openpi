"""Count parameter shapes and trainable subsets without materializing model weights."""
import argparse
import json
from pathlib import Path

import flax.nnx as nnx
import jax

from openpi.shared import nnx_utils
from openpi.training import config as config_lib


def count(state):
    total=0
    for leaf in jax.tree.leaves(state):
        shape=getattr(leaf,'shape',None)
        if shape is None:continue
        n=1
        for d in shape:n*=int(d)
        total+=n
    return total


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='pi05_robocasa_spatial_query_human4');p.add_argument('--output',type=Path)
    a=p.parse_args();cfg=config_lib.get_config(a.config)
    model=nnx.eval_shape(lambda:cfg.model.create(jax.random.key(0)))
    state=nnx.state(model)
    trainable=state.filter(cfg.trainable_filter)
    frozen=state.filter(cfg.freeze_filter)
    query=state.filter(nnx_utils.PathRegex('.*spatial_query.*'))
    head=state.filter(nnx_utils.PathRegex('.*geometry_head.*'))
    lora=state.filter(nnx_utils.PathRegex('.*lora.*'))
    lora_trainable=state.filter(nnx.All(cfg.trainable_filter,nnx_utils.PathRegex('.*lora.*')))
    query_trainable=state.filter(nnx.All(cfg.trainable_filter,nnx_utils.PathRegex('.*spatial_query.*')))
    head_trainable=state.filter(nnx.All(cfg.trainable_filter,nnx_utils.PathRegex('.*geometry_head.*')))
    result={'config':cfg.name,'method':'NNX abstract parameter shapes; no model weights materialized',
        'total_parameters':count(state),'trainable_parameters':count(trainable),'frozen_parameters':count(frozen),
        'spatial_query_parameters':count(query),'spatial_query_trainable_parameters':count(query_trainable),
        'geometry_head_parameters':count(head),'geometry_head_trainable_parameters':count(head_trainable),
        'lora_parameters':count(lora),'lora_trainable_parameters':count(lora_trainable),
        'action_projections_trainable_parameters':count(state.filter(nnx.All(cfg.trainable_filter,nnx_utils.PathRegex('.*(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out).*')))),
        'note':'Only Action-Expert LoRA, SpatialQuery, and GeometryHead parameters are expected to train.'}
    if result['spatial_query_trainable_parameters']!=result['spatial_query_parameters'] or result['geometry_head_trainable_parameters']!=result['geometry_head_parameters']:
        raise RuntimeError('SpatialQuery or GeometryHead was unexpectedly frozen')
    if result['lora_parameters'] and result['lora_trainable_parameters']==0:
        raise RuntimeError('all LoRA parameters are frozen; inspect the checkpoint/config parameter paths')
    unexpected=state.filter(nnx.All(cfg.trainable_filter,nnx.Not(nnx_utils.PathRegex('.*(lora|spatial_query|geometry_head).*'))))
    if count(unexpected):
        raise RuntimeError(f"unexpected trainable parameters outside LoRA/SpatialQuery/GeometryHead: {count(unexpected)}")
    if result['action_projections_trainable_parameters']:
        raise RuntimeError('pretrained action projections must remain frozen for the LoRA-only config')
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
