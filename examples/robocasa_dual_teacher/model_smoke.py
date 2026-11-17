"""Real checkpoint + real Human + real RoMa/EE target gradient checks; no optimizer update."""
import dataclasses,json,os
from pathlib import Path
import flax.nnx as nnx
import jax,jax.numpy as jnp
import numpy as np,optax
from openpi.shared import nnx_utils
from openpi.training import config as config_lib,data_loader
from openpi.models import model as model_lib

def main():
 cfg=config_lib.get_config('pi05_robocasa_roma_ee_dual');task='PickPlaceSinkToCounter';episode=86;frame=0
 root=Path('/data4/xuchenfei/robocasa365/data/household_augmented');cache=Path('/data4/xuchenfei/robocasa365/output/roma_ee_dual_teacher/smoke/cache_v1');output=Path('/data4/xuchenfei/robocasa365/output/roma_ee_dual_teacher/smoke/model_validation.json')
 meta=next(x for x in cfg.data.data_dirs if x['task']==task).copy();meta['path']=str(root/task/'lerobot')
 cfg=dataclasses.replace(cfg,data=dataclasses.replace(cfg.data,data_dirs=[meta],spatial_teacher_cache_dir=str(cache),minimum_teacher_episode_coverage=0),batch_size=1,num_workers=0)
 dc=cfg.data.create(cfg.assets_dirs,cfg.model);ds=data_loader.create_torch_dataset(dc,cfg.model.action_horizon,cfg.model)
 def fixed(i):
  ds._last_sample=(ds.datasets[0],episode,frame);return ds._last_sample
 ds.sample_step=fixed
 batch=data_loader.transform_dataset(ds,dc)[0]
 batch=jax.tree.map(lambda x:jnp.asarray(x)[None],batch)
 observation=model_lib.Observation.from_dict(batch);actions=batch['actions'];teacher=batch['teacher_latent'];valid=batch['teacher_valid'];scene=teacher[:,:64];sv=valid[:,:64];ee=teacher[:,64:];ev=valid[:,64:]
 if int(sv.sum())==0 or int(ev.sum())!=6:raise ValueError('Real teacher has insufficient valid cells')
 print('real data loaded',actions.shape,int(sv.sum()),int(ev.sum()),flush=True)
 rng=jax.random.key(cfg.seed);model0=cfg.model.create(rng);_,initial=nnx.split(model0);params=cfg.weight_loader.load(initial.to_pure_dict());model=cfg.model.load(params);del model0,initial,params
 rng=jax.random.fold_in(rng,71)
 pattern='.*(spatial_query|scene_hidden|scene_head|ee_hidden|ee_head|lora).*';diff=nnx.DiffState(0,nnx_utils.PathRegex(pattern));result={'checkpoint':cfg.weight_loader.params_path,'sample':{'task':task,'episode':episode,'frame':frame},'action_shape':list(actions.shape),'scene_valid':int(sv.sum()),'ee_valid':int(ev.sum()),'test_scope':'real checkpoint, real RGB/state/action and real RoMa/MuJoCo teachers; gradients only, no optimizer step','device':str(jax.devices()[0])}
 for index,name in enumerate(['action','scene','ee']):
  def fn(m):return m.compute_dual_losses(rng,observation,actions,scene,sv,ee,ev,train=False)[index]
  loss,grads=nnx.value_and_grad(fn,argnums=diff)(model)
  metrics={'loss':float(loss)}
  for label,reg in [('query','.*spatial_query.*'),('scene_head','.*(scene_hidden|scene_head).*'),('ee_head','.*(ee_hidden|ee_head).*'),('lora','.*lora.*')]:
   metrics[label+'_gradient_norm']=float(optax.global_norm(grads.filter(nnx_utils.PathRegex(reg))))
  result[name]=metrics;print(name,metrics,flush=True)
 for name,key in [('action','query'),('action','lora'),('scene','query'),('scene','scene_head'),('ee','query'),('ee','ee_head')]:
  val=result[name][key+'_gradient_norm']
  if not np.isfinite(val) or val<=0:raise RuntimeError(f'{name} {key} gradient is not finite/nonzero: {val}')
 sample=model.sample_actions(jax.random.fold_in(rng,1),observation,num_steps=2)
 if sample.shape!=(1,50,32) or not bool(jnp.isfinite(sample).all()):raise RuntimeError('Invalid teacher-free action sampling')
 result['sample_shape']=list(sample.shape);result['status']='passed';output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
