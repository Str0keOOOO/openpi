"""CPU-only bounded search for rejected starts; fixed scenes, Anchor, sign and magnitude strata."""
import argparse,gzip,json,sys,time,traceback
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import differential_evolution
from scipy.stats import qmc
import mujoco
from grouped import dump,digest,MODEL_ARRAYS,validate_delta

class Found(Exception):pass

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--task',required=True);p.add_argument('--scene',type=int,required=True);p.add_argument('--sobol',type=int,default=256);p.add_argument('--iterations',type=int,default=50);p.add_argument('--resample',action='store_true');a=p.parse_args();began=time.time();src=a.dataset/'scenes'/a.task/f'scene_{a.scene:02d}';out=a.output/'scenes'/a.task/f'scene_{a.scene:02d}';out.mkdir(parents=True,exist_ok=True)
 try:
  from robocasa.environments.kitchen.kitchen import Kitchen
  c=json.loads((a.dataset/'config.json').read_text());anchor=json.loads((src/'anchor/anchor.json').read_text());remapper=Kitchen.__new__(Kitchen);remapper._cam_configs={};remapper.generative_textures=None;xml=remapper.edit_model_xml(gzip.open(src/'anchor/model.xml.gz','rt').read());m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
  with np.load(src/'anchor/snapshot.npz') as z:base={k:z[k].copy() for k in z.files}
  for key in MODEL_ARRAYS:getattr(m,key)[:]=base[key]
  def reset():
   d.time=base['state'][0];d.qpos[:]=base['qpos'];d.qvel[:]=base['qvel'];d.ctrl[:]=base['ctrl'];mujoco.mj_forward(m,d)
  reset();bid=m.body('mobilebase0_base').id;bp=d.xpos[bid].copy();r0=d.xmat[bid].reshape(3,3).copy();yaw0=Rotation.from_matrix(r0).as_euler('xyz')[2];basis=np.array([[np.cos(yaw0),-np.sin(yaw0)],[np.sin(yaw0),np.cos(yaw0)]]);yawid=m.joint('mobilebase0_joint_mobile_yaw').id;ya=int(m.jnt_qposadr[yawid]);pivot=d.xanchor[yawid].copy();xy=[int(m.jnt_qposadr[m.joint(n).id]) for n in ['mobilebase0_joint_mobile_forward','mobilebase0_joint_mobile_side']];J=np.zeros((2,2))
  for i,addr in enumerate(xy):
   reset();d.qpos[addr]+=1e-4;mujoco.mj_forward(m,d);J[:,i]=(d.xpos[bid,:2]-bp[:2])/1e-4
  reset();d.qpos[ya]+=1e-4;mujoco.mj_forward(m,d);yaw_jac=((Rotation.from_matrix(d.xmat[bid].reshape(3,3)).as_euler('xyz')[2]-yaw0+np.pi)%(2*np.pi)-np.pi)/1e-4;reset()
  if abs(yaw_jac-1)>1e-5:raise RuntimeError('Unexpected yaw joint orientation')
  Ji=np.linalg.inv(J);names=['mobilebase0_joint_torso_height']+[f'robot0_joint{i}' for i in range(1,8)];ids=[m.joint(n).id for n in names];addrs=[int(m.jnt_qposadr[i]) for i in ids];robotids={i for i in range(m.ngeom) if (m.geom(i).name or '').startswith(('robot0_','mobilebase0_','gripper0_'))}
  def realize(delta):
   d.time=base['state'][0];d.qpos[:]=base['qpos'];d.qvel[:]=base['qvel'];d.ctrl[:]=base['ctrl'];angle=np.radians(delta[2]);R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);after=pivot[:2]+R@(bp[:2]-pivot[:2]);target=bp[:2]+basis@delta[:2];d.qpos[xy]=base['qpos'][xy]+Ji@(target-after);d.qpos[ya]+=angle;d.qpos[addrs]+=np.r_[delta[3],np.radians(delta[4:])];mujoco.mj_forward(m,d)
   actual=np.r_[basis.T@(d.xpos[bid,:2]-bp[:2]),np.degrees((Rotation.from_matrix(d.xmat[bid].reshape(3,3)).as_euler('xyz')[2]-yaw0+np.pi)%(2*np.pi)-np.pi),d.qpos[addrs[0]]-base['qpos'][addrs[0]],np.degrees(d.qpos[addrs[1:]]-base['qpos'][addrs[1:]])]
   if not np.allclose(actual,delta,atol=1e-7,rtol=0):raise RuntimeError('Analytic base placement disagrees with actual MuJoCo pose')
   return actual
  def contacts():return [x for x in d.contact if (int(x.geom1) in robotids)!=(int(x.geom2) in robotids)]
  def limits():return any(m.jnt_limited[i] and not m.jnt_range[i,0]<=d.qpos[addr]<=m.jnt_range[i,1] for i,addr in zip(ids,addrs))
  # Reconstruct independently accepted starts as a CPU geometry cross-check.
  checked=0
  for file in sorted((src/'starts').glob('*/configuration.json')):
   v=json.loads(file.read_text())
   if not v['accepted']:continue
   realize(np.asarray(v['actual_delta']))
   with np.load(v['state_file']) as z:
    if np.max(np.abs(d.qpos-z['qpos']))>1e-7 or contacts() or limits():raise RuntimeError('CPU geometry did not reproduce an official accepted start')
   checked+=1
  dump(out/'cpu_geometry_validation.json',dict(passed=True,accepted_starts_checked=checked,policy_actions_executed=0))
  if a.resample:
   from legal_resampling import resample_missing
   resample_missing(c,src,out,realize,contacts,limits,m,d,base)
   return
  summaries=[]
  for file in sorted((src/'starts').glob('*/configuration.json')):
   record=json.loads(file.read_text())
   if record['accepted']:continue
   case=record['case'];dest=out/case['id'];dest.mkdir(exist_ok=True);marker=dest/'search_result.json'
   if marker.exists():summaries.append(json.loads(marker.read_text()));continue
   active=case['active_dimensions'];tier=['small','medium','large'].index(case['level']);lo,hi=c['level_multipliers'][tier];bounds=[]
   for axis in active:
    width=(hi-lo)*c['training_limits'][axis]/5;low=lo*c['training_limits'][axis]+case['magnitude_subbins'][axis]*width;high=low+width;low=max(low,np.nextafter(c['tolerances'][axis],np.inf));bounds.append((low,high))
   lower=np.array([x[0] for x in bounds]);upper=np.array([x[1] for x in bounds]);seed=int(digest([c['seed'],a.task,a.scene,case['id'],'deep_cpu'])[:8],16);best=[float('inf'),None,None];calls=[0]
   def objective(u):
    delta=np.zeros(11);delta[active]=(lower+np.asarray(u)*(upper-lower))*np.asarray(case['signs'])[active];actual=realize(delta);coll=contacts();calls[0]+=1
    value=0. if not coll and not limits() else 1.+1000*max([max(-float(x.dist),0.) for x in coll]+[0.])+.001*len(coll)
    if value<best[0]:best[:]=[value,delta.copy(),actual.copy()]
    if value==0:
     if not validate_delta(c,case,actual):raise RuntimeError('Candidate left its original stratum')
     state=np.r_[d.time,d.qpos,d.qvel]
     if len(state)!=len(base['state']):raise RuntimeError('Unexpected activation state layout')
     np.savez_compressed(dest/'candidate_state.npz',state=state,qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),**{key:np.asarray(getattr(m,key)).copy() for key in MODEL_ARRAYS});raise Found()
    return value
   found=False;start=time.time()
   try:
    # Corners plus low-discrepancy coverage, followed by geometric minimization.
    objective(np.full(len(active),.5));sampler=qmc.Sobol(d=len(active),scramble=True,seed=seed)
    for point in sampler.random_base2(int(np.ceil(np.log2(a.sobol)))):objective(point)
    differential_evolution(objective,[(0.,1.)]*len(active),seed=seed,popsize=5,maxiter=a.iterations,polish=False,tol=1e-7,atol=1e-7,updating='immediate',workers=1)
   except Found:found=True
   realize(best[1]);pairs=[dict(geoms=[m.geom(int(x.geom1)).name,m.geom(int(x.geom2)).name],distance_m=float(x.dist)) for x in contacts()]
   summary=dict(task=a.task,scene_index=a.scene,id=case['id'],case=case,status='legal_candidate_pending_official_export' if found else 'not_found_in_original_stratum',evaluations=calls[0],seconds=time.time()-start,best_objective=best[0],actual_delta=best[2].tolist(),best_contacts=pairs,original_anchor_unchanged=True,original_tier_sign_subbin_unchanged=True,policy_actions_executed=0);dump(marker,summary);summaries.append(summary);print(a.task,a.scene,case['id'],summary['status'],calls[0],flush=True)
  dump(out/'summary.json',dict(task=a.task,scene_index=a.scene,missing=len(summaries),found=sum(x['status']=='legal_candidate_pending_official_export' for x in summaries),not_found=sum(x['status']!='legal_candidate_pending_official_export' for x in summaries),seconds=time.time()-began,policy_actions_executed=0))
 except Exception:dump(out/'summary.json',dict(task=a.task,scene_index=a.scene,status='error',error=traceback.format_exc()));print(traceback.format_exc(),flush=True)
if __name__=='__main__':main()
