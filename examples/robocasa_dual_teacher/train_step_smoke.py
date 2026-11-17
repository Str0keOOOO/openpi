"""One isolated optimizer step using real observations and real dual teachers."""
import dataclasses,functools,json
from pathlib import Path
import flax.nnx as nnx
import jax,jax.numpy as jnp
import numpy as np
from openpi.training import config as config_lib,data_loader,optimizer,utils
from openpi.models import model as model_lib
from openpi.shared import nnx_utils
from scripts.train import train_step
from examples.robocasa_spatial.validate_optimizer_roundtrip import _group_delta

def main():
 cfg=config_lib.get_config('pi05_robocasa_roma_ee_dual');task='PickPlaceSinkToCounter';episode=86
 root=Path('/data4/xuchenfei/robocasa365/data/household_augmented');cache=Path('/data4/xuchenfei/robocasa365/output/roma_ee_dual_teacher/smoke/cache_v1');out=Path('/data4/xuchenfei/robocasa365/output/roma_ee_dual_teacher/smoke/train_step_validation.json')
 meta=next(x for x in cfg.data.data_dirs if x['task']==task).copy();meta['path']=str(root/task/'lerobot')
 cfg=dataclasses.replace(cfg,data=dataclasses.replace(cfg.data,data_dirs=[meta],spatial_teacher_cache_dir=str(cache),minimum_teacher_episode_coverage=0),batch_size=1,num_workers=0)
 dc=cfg.data.create(cfg.assets_dirs,cfg.model);ds=data_loader.create_torch_dataset(dc,cfg.model.action_horizon,cfg.model)
 def fixed(i):ds._last_sample=(ds.datasets[0],episode,0);return ds._last_sample
 ds.sample_step=fixed;raw=data_loader.transform_dataset(ds,dc)[0];raw=jax.tree.map(lambda x:jnp.asarray(x)[None],raw)
 obs=model_lib.Observation.from_dict(raw);t=raw['teacher_latent'];v=raw['teacher_valid'];batch=(obs,raw['actions'],t[:,:64],v[:,:64],t[:,64:],v[:,64:])
 if int(v[:,:64].sum())==0 or int(v[:,64:].sum())!=6:raise RuntimeError('Expected real teacher sample')
 rng=jax.random.key(cfg.seed);m0=cfg.model.create(rng);_,initial=nnx.split(m0);loaded=cfg.weight_loader.load(initial.to_pure_dict());model=cfg.model.load(loaded);del m0,initial,loaded
 params=nnx_utils.state_map(nnx.state(model),cfg.freeze_filter,lambda p:p.replace(p.value.astype(jnp.bfloat16)))
 trainable=params.filter(cfg.trainable_filter);frozen=params.filter(cfg.freeze_filter)
 total_trainable=sum(x.size for x in jax.tree.leaves(trainable));total_frozen=sum(x.size for x in jax.tree.leaves(frozen));print('params',total_trainable,total_frozen,flush=True)
 tx=optimizer.create_optimizer(cfg.optimizer,cfg.lr_schedule,weight_decay_mask=None)
 state=utils.TrainState(step=jnp.asarray(0,jnp.int32),params=params,model_def=nnx.graphdef(model),tx=tx,opt_state=tx.init(trainable),ema_decay=None,ema_params=None)
 before_frozen=np.asarray(jax.device_get(jax.tree.leaves(frozen)[0])).reshape(-1)[:64].copy()
 updated,metrics=jax.jit(functools.partial(train_step,cfg))(rng,state,batch);jax.block_until_ready(metrics)
 values={k:float(jax.device_get(v)) for k,v in metrics.items()};deltas={};counts={}
 for marker in ['spatial_query','scene_hidden','scene_head','ee_hidden','ee_head','lora']:
  deltas[marker],counts[marker]=_group_delta(state.params,updated.params,marker)
  if counts[marker]<=0 or not np.isfinite(deltas[marker]) or deltas[marker]<=0:raise RuntimeError(f'No finite parameter update for {marker}: {deltas[marker]}')
 after=np.asarray(jax.device_get(jax.tree.leaves(updated.params.filter(cfg.freeze_filter))[0])).reshape(-1)[:64]
 if not np.array_equal(before_frozen,after):raise RuntimeError('Frozen parameter changed')
 if int(updated.step)!=1 or not all(np.isfinite(x) for x in values.values()):raise RuntimeError('Invalid optimizer step')
 result=dict(status='passed',scope='one isolated real-data optimizer step; no formal training or checkpoint',losses=values,parameter_delta_l2=deltas,parameter_counts=counts,trainable_parameters=total_trainable,frozen_parameters=total_frozen,frozen_probe_unchanged=True,formal_training_started=False)
 out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
