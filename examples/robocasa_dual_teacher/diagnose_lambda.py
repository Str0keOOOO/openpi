"""One real-sample loss/Query-gradient diagnostic, without optimizer updates."""
import argparse,dataclasses,json,os,time
from pathlib import Path
import flax.nnx as nnx
import jax,jax.numpy as jnp
import numpy as np
from openpi.training import config as config_lib,data_loader
from openpi.models import model as model_lib
from openpi.shared import nnx_utils
from examples.robocasa_dual_teacher.assemble_cache import assemble

def main():
 p=argparse.ArgumentParser();p.add_argument('--window',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 started=time.time();meta=json.loads(str(np.load(a.window,allow_pickle=False)['metadata']));task=meta['task'];episode=int(meta['episode']);frame=int(meta['frame']);root=Path(meta['dataset'])
 assemble([a.window],a.output/'diagnostic_cache');os.environ['OPENPI_DUAL_TEACHER_PARTIAL_SMOKE']='1'
 cfg=config_lib.get_config('pi05_robocasa_roma_ee_dual');entry=next(x for x in cfg.data.data_dirs if x['task']==task).copy();entry['path']=str(root/task/'lerobot')
 cfg=dataclasses.replace(cfg,data=dataclasses.replace(cfg.data,data_dirs=[entry],spatial_teacher_cache_dir=str(a.output/'diagnostic_cache'),minimum_teacher_episode_coverage=0),batch_size=1,num_workers=0)
 dc=cfg.data.create(cfg.assets_dirs,cfg.model);ds=data_loader.create_torch_dataset(dc,cfg.model.action_horizon,cfg.model)
 def fixed(i):ds._last_sample=(ds.datasets[0],episode,frame);return ds._last_sample
 ds.sample_step=fixed;raw=data_loader.transform_dataset(ds,dc)[0];raw=jax.tree.map(lambda x:jnp.asarray(x)[None],raw)
 obs=model_lib.Observation.from_dict(raw);actions=raw['actions'];t=raw['teacher_latent'];v=raw['teacher_valid'];assert int(v[:,:64].sum())>0 and int(v[:,64:].sum())==6
 print('REAL_SAMPLE',task,episode,frame,'ACTIONS',actions.shape,'TEACHERS',int(v[:,:64].sum()),int(v[:,64:].sum()),flush=True)
 rng=jax.random.key(cfg.seed);m0=cfg.model.create(rng);_,initial=nnx.split(m0);loaded=cfg.weight_loader.load(initial.to_pure_dict());model=cfg.model.load(loaded);del m0,initial,loaded
 # Match training's frozen-backbone dtype; new modules and LoRA remain trainable dtype.
 params=nnx_utils.state_map(nnx.state(model),cfg.freeze_filter,lambda p:p.replace(p.value.astype(jnp.bfloat16)))
 graph=nnx.graphdef(model);q=nnx.state(model).spatial_query.value;rng=jax.random.fold_in(rng,71)
 @jax.jit
 def evaluate(query,state,ob,act,teacher,valid,key):
  def losses(query_value):
   m=nnx.merge(graph,state);m.spatial_query.value=query_value
   values=m.compute_dual_losses(key,ob,act,teacher[:,:64],valid[:,:64],teacher[:,64:],valid[:,64:],train=False)
   return jnp.stack(values[:3])
  values,pullback=jax.vjp(losses,query)
  grads=jax.lax.map(lambda basis:pullback(basis)[0],jnp.eye(3,dtype=values.dtype))
  return values,grads
 print('CHECKPOINT_LOADED; compiling one forward and three Query-gradient probes',flush=True)
 losses,grads=evaluate(q,params,obs,actions,t,v,rng);jax.block_until_ready(losses);losses=np.asarray(losses,dtype=float);grads=np.asarray(grads,dtype=float).reshape(3,-1)
 assert np.isfinite(losses).all() and np.isfinite(grads).all();norms=np.linalg.norm(grads,axis=1);assert (norms>0).all()
 cases=[]
 for ls,le in [(.05,.01),(.1,.1),(.1,.01),(.1,.05)]:
  weighted_scene=ls*losses[1];weighted_ee=le*losses[2];aux=ls*grads[1]+le*grads[2]
  cases.append(dict(lambda_scene=ls,lambda_ee=le,weighted_scene_loss=weighted_scene,weighted_ee_loss=weighted_ee,total_loss=losses[0]+weighted_scene+weighted_ee,scene_to_action_loss_ratio=weighted_scene/losses[0],ee_to_action_loss_ratio=weighted_ee/losses[0],aux_to_action_loss_ratio=(weighted_scene+weighted_ee)/losses[0],scene_to_action_query_gradient_ratio=ls*norms[1]/norms[0],ee_to_action_query_gradient_ratio=le*norms[2]/norms[0],combined_aux_to_action_query_gradient_ratio=float(np.linalg.norm(aux)/norms[0]),scene_query_gradient_cosine_to_action=float(grads[0]@grads[1]/(norms[0]*norms[1])),ee_query_gradient_cosine_to_action=float(grads[0]@grads[2]/(norms[0]*norms[2]))))
 result=dict(status='passed',sample=dict(task=task,episode=episode,frame=frame,anchor_episode=meta['anchor_episode'],dataset=str(root)),batch_size=1,seed=cfg.seed,flow_matching_seed_fold=71,checkpoint=cfg.weight_loader.params_path,losses=dict(zip(['action','scene','ee'],losses.tolist())),query_gradient_norms=dict(zip(['action','scene','ee'],norms.tolist())),weights=cases,scene_valid_dims=int(v[:,:64].sum()),ee_valid_dims=int(v[:,64:].sum()),scope='Single real large-offset sample, Base checkpoint with newly initialized Query/heads; diagnostic only, not multitask calibrated weights',optimizer_updates=0,elapsed_seconds=time.time()-started)
 (a.output/'result.json').write_text(json.dumps(result,indent=2));np.savez_compressed(a.output/'loss_query_gradients.npz',losses=losses,query_gradients=grads);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
