"""Real final-cache smoke tests; no formal checkpoint or training budget used."""
import argparse,dataclasses,functools,json,os,time
from pathlib import Path
os.environ['OPENPI_DENSE_TEACHER_SMOKE']='1'
import flax.nnx as nnx
import jax,jax.numpy as jnp,numpy as np
from openpi.training import config as configs,data_loader,optimizer,utils
from openpi.models import model as model_lib
from openpi.shared import nnx_utils
from scripts.train import train_step
from examples.robocasa_spatial.validate_optimizer_roundtrip import _group_delta
from examples.robocasa_dual_teacher.cache_schema import VERSION,validate_metadata

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['gradients','optimizer','inference'],required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--task',default='PickPlaceSinkToCounter');p.add_argument('--episode',type=int,default=499);p.add_argument('--frame',type=int,default=0)
 a=p.parse_args();start=time.time()
 cfg=configs.get_config('pi05_household_dual_teacher_30k');task=a.task;episode=a.episode
 meta=json.loads((Path(cfg.data.spatial_teacher_cache_dir)/task/f'episode_{episode:06d}'/'metadata.json').read_text());validate_metadata(meta)
 entry=next(x for x in cfg.data.data_dirs if x['task']==task)
 cfg=dataclasses.replace(cfg,data=dataclasses.replace(cfg.data,data_dirs=[entry]),batch_size=1,num_workers=0)
 dc=cfg.data.create(cfg.assets_dirs,cfg.model);ds=data_loader.create_torch_dataset(dc,cfg.model.action_horizon,cfg.model)
 def fixed(i):ds._last_sample=(ds.datasets[0],episode,a.frame);return ds._last_sample
 ds.sample_step=fixed;raw=data_loader.transform_dataset(ds,dc)[0];raw=jax.tree.map(lambda x:jnp.asarray(x)[None],raw)
 ob=model_lib.Observation.from_dict(raw);act=raw['actions'];f=raw['scene_flow'];v=raw['scene_pixel_valid'];ee=raw['ee_teacher'];ev=raw['ee_teacher_valid'];batch=(ob,act,f,v,ee,ev)
 if f.shape!=(1,2,256,256,2) or int(v.sum())==0 or int(ev.sum())!=6:raise ValueError('Missing real teachers')
 rng=jax.random.key(cfg.seed);m0=cfg.model.create(rng);_,initial=nnx.split(m0);loaded=cfg.weight_loader.load(initial.to_pure_dict());m=cfg.model.load(loaded);del m0,initial,loaded
 state=nnx_utils.state_map(nnx.state(m),cfg.freeze_filter,lambda p:p.replace(p.value.astype(jnp.bfloat16)));graph=nnx.graphdef(m);del m
 result=dict(stage=a.stage,config=cfg.name,sample=dict(task=task,episode=episode,frame=a.frame),checkpoint=cfg.weight_loader.params_path,real_rgb=True,real_dense_roma=True,native_image_augmentation=True,formal_updates=0,cache_version=VERSION,roma_setting=meta['roma_setting'],roma_bidirectional=meta['roma_bidirectional'],lambda_scene=cfg.lambda_scene,lambda_ee=cfg.lambda_ee)
 if a.stage=='gradients':
  @jax.jit
  def probe(query,params):
   def losses(q):
    model=nnx.merge(graph,params);model.spatial_query.value=q
    return jnp.stack(model.compute_dual_losses(rng,ob,act,f,v,ee,ev,train=True)[:3])
   values,pb=jax.vjp(losses,query);g=jax.lax.map(lambda x:pb(x)[0],jnp.eye(3));return values,g
  values,g=probe(state.spatial_query.value,state);jax.block_until_ready(g);values=np.asarray(values);g=np.asarray(g).reshape(3,-1);norm=np.linalg.norm(g,axis=1)
  if not np.isfinite(values).all() or not np.isfinite(norm).all() or not (norm>0).all():raise ValueError('Invalid/zero individual Query gradients')
  result.update(losses=dict(zip(['action','scene','ee'],map(float,values))),query_gradient_norms=dict(zip(['action','scene','ee'],map(float,norm))),weighted_aux_to_action_loss=float(.1*(values[1]+values[2])/values[0]))
 elif a.stage=='optimizer':
  tx=optimizer.create_optimizer(cfg.optimizer,cfg.lr_schedule,weight_decay_mask=None);ts=utils.TrainState(step=jnp.asarray(0,jnp.int32),params=state,model_def=graph,tx=tx,opt_state=tx.init(state.filter(cfg.trainable_filter)),ema_decay=None,ema_params=None)
  frozen=state.filter(cfg.freeze_filter);before=np.asarray(jax.tree.leaves(frozen)[0]).reshape(-1)[:64].copy()
  snapshot={}
  for path,value in jax.tree_util.tree_flatten_with_path(ts.params.to_pure_dict())[0]:
   key=jax.tree_util.keystr(path)
   if any(k in key for k in ['spatial_query','scene_hidden','scene_head','ee_hidden','ee_head','lora']):snapshot[key]=np.asarray(value).copy()
  updated,metrics=jax.jit(functools.partial(train_step,cfg),donate_argnums=(1,))(rng,ts,batch);jax.block_until_ready(metrics);values={k:float(x) for k,x in metrics.items()};deltas={};counts={}
  current={jax.tree_util.keystr(p):v for p,v in jax.tree_util.tree_flatten_with_path(updated.params.to_pure_dict())[0]}
  for marker in ['spatial_query','scene_hidden','scene_head','ee_hidden','ee_head','lora']:
   items=[k for k in snapshot if marker in k];counts[marker]=sum(snapshot[k].size for k in items);deltas[marker]=float(np.sqrt(sum(np.sum((np.asarray(current[k],dtype=np.float64)-snapshot[k])**2) for k in items)))
   if counts[marker]<=0 or not np.isfinite(deltas[marker]) or deltas[marker]<=0:raise ValueError(f'No parameter update: {marker}')
  after=np.asarray(jax.tree.leaves(updated.params.filter(cfg.freeze_filter))[0]).reshape(-1)[:64]
  if not np.array_equal(before,after) or int(updated.step)!=1 or not all(np.isfinite(x) for x in values.values()):raise ValueError('Invalid isolated optimizer update')
  result.update(metrics=values,parameter_deltas=deltas,parameter_counts=counts,frozen_probe_unchanged=True,isolated_diagnostic_updates=1,trainable_parameters=sum(x.size for x in jax.tree.leaves(state.filter(cfg.trainable_filter))),frozen_parameters=sum(x.size for x in jax.tree.leaves(frozen)))
 else:
  @jax.jit
  def sample(params):
   model=nnx.merge(graph,params);return model.sample_actions(rng,ob,num_steps=2)
  action=sample(state);jax.block_until_ready(action)
  changed=state.to_pure_dict();changed['spatial_query']=np.zeros_like(changed['spatial_query']);altered=cfg.model.load(changed);altered_state=nnx.state(altered);del altered
  other=sample(altered_state);delta=float(jnp.max(jnp.abs(other-action)))
  if action.shape!=(1,50,32) or not bool(jnp.isfinite(action).all()) or not np.isfinite(delta) or delta<=0:raise ValueError('Invalid teacher-free sampling or Query has no effect')
  result.update(action_shape=list(action.shape),query_ablation_max_action_delta=delta,inference_inputs='three RGB + State + language only')
 result.update(status='passed',elapsed_seconds=time.time()-start);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
