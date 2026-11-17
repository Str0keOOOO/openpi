"""Restore one genuine recorded window and paired Anchor for offline teacher prep.

MuJoCo produces masks and mask/RGB alignment checks, never EE labels or flow. RGBs come from the
recorded videos. Outputs are temporary teacher inputs, not altered datasets.
"""
import argparse,csv,gzip,hashlib,json,functools
from pathlib import Path
import cv2,numpy as np
from scipy.spatial.transform import Rotation
from examples.robocasa_spatial.export_dynamic_masks import (_open_env,_geom_id_image,_robot_body_ids,_episode_moved_geoms)
from examples.robocasa_spatial.generate_teacher_cache import robot_geom_ids
from examples.robocasa_dual_teacher.state_ee import episode_teacher,EE_VERSION,EE_FILE
CAMS=('robot0_agentview_left','robot0_agentview_right')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
@functools.lru_cache(maxsize=4)
def _read_states_cached(path, size, mtime_ns):
 with np.load(path,allow_pickle=False) as z:return z['states'].copy()
def read_states(path):
 p=Path(path);st=p.stat();return _read_states_cached(str(p),st.st_size,st.st_mtime_ns)
def rgb(ds,episode,camera,frame):
 p=ds/'videos'/f'chunk-{episode//1000:03d}'/f'observation.images.{camera}'/f'episode_{episode:06d}.mp4'
 cap=cv2.VideoCapture(str(p))
 try:
  if not cap.isOpened() or not cap.set(cv2.CAP_PROP_POS_FRAMES,frame):raise RuntimeError(f'Cannot seek video {p}')
  ok,bgr=cap.read()
  if not ok:raise RuntimeError(f'Cannot decode {p} frame {frame}')
  return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),p
 finally:cap.release()
def transform(pos,rot):
 t=np.eye(4);t[:3,:3]=np.asarray(rot).reshape(3,3);t[:3,3]=pos;return t

def main(argv=None, *, return_arrays=False):
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--task',required=True);ap.add_argument('--episode',type=int,required=True);ap.add_argument('--frame',type=int,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--position-scale-m',type=float,default=.1);ap.add_argument('--rotation-scale-rad',type=float,default=1.);ap.add_argument('--mask-policy',choices=['robot_only','robot_and_dynamic'],default='robot_only');a=ap.parse_args(argv)
 ds=a.dataset/a.task/'lerobot';ep=ds/'extras'/f'episode_{a.episode:06d}';prov=json.loads((ep/'provenance.json').read_text());typ=prov.get('data_type',prov.get('type'))
 if typ=='anchor':anchor=a.episode
 elif typ in ('perturb_single','perturb_multi'):
  anchor=prov.get('reference_anchor_episode_index')
 else:
  pairs=[r for r in csv.DictReader((a.dataset/'human_anchor_pairs.csv').open()) if r['task']==a.task and int(r['human_episode_index'])==a.episode]
  if len(pairs)!=1:raise ValueError('Human to Anchor provenance must be unique')
  anchor=int(pairs[0]['anchor_episode_index'])
 if anchor is None:raise ValueError('Missing actual paired Anchor reference')
 anchor=int(anchor);ae=ds/'extras'/f'episode_{anchor:06d}';aprov=json.loads((ae/'provenance.json').read_text())
 if aprov.get('data_type',aprov.get('type'))!='anchor':raise ValueError('Reference is not an Anchor')
 if typ!='anchor' and int(aprov['source_human_episode_index'])!=int(prov.get('source_human_episode_index',a.episode)):raise ValueError('Human and Anchor source mismatch')
 matches=[r for r in csv.DictReader((a.dataset/'human_anchor_pairs.csv').open()) if r['task']==a.task and int(r['anchor_episode_index'])==anchor]
 if len(matches)!=1 or int(matches[0]['human_episode_index'])!=int(aprov['source_human_episode_index']):raise ValueError('Anchor pair index is inconsistent')
 anchor_method=matches[0].get('anchor_generation_method','not_recorded')
 if gzip.open(ep/'model.xml.gz','rb').read()!=gzip.open(ae/'model.xml.gz','rb').read():raise ValueError('Current and Anchor XML differ; cannot assume geom IDs match')
 states=read_states(ep/'states.npz');astates=read_states(ae/'states.npz')
 if not 0<=a.frame<len(states):raise IndexError('Frame out of bounds')
 meta=json.loads((ep/'ep_meta.json').read_text());dm=json.loads((ds/'extras/dataset_meta.json').read_text())
 env=_open_env(a.task,a.episode,ds,meta,dm,states)
 try:
  sim=env.sim; rg=robot_geom_ids(sim); rb=_robot_body_ids(sim,rg)
  dynamic=np.empty(0,dtype=int)
  if a.mask_policy=='robot_and_dynamic':
   gm,_=_episode_moved_geoms(sim,states,rb,.003,3.);gam,_=_episode_moved_geoms(sim,astates,rb,.003,3.)
   sim.set_state_from_flattened(states[0]);sim.forward();p0=np.asarray(sim.data.body_xpos).copy();q0=np.asarray(sim.data.body_xquat).copy()
   sim.set_state_from_flattened(astates[0]);sim.forward();pa=np.asarray(sim.data.body_xpos).copy();qa=np.asarray(sim.data.body_xquat).copy()
   moved=(np.linalg.norm(p0-pa,axis=1)>.003)|(2*np.arccos(np.clip(np.abs(np.sum(q0*qa,axis=1)),0,1))>np.deg2rad(3))
   ids=[]
   for gid in range(sim.model.ngeom):
    bid=int(sim.model.geom_bodyid[gid]);name=sim.model.body_id2name(bid) or ''
    if bid not in rb and (moved[bid] or name.startswith('obj_')):ids.append(gid)
   dynamic=np.unique(np.concatenate([gm,gam,np.asarray(ids,dtype=int)]))
  excluded=np.unique(np.concatenate([np.asarray(rg,dtype=int),dynamic]))
  sim.set_state_from_flattened(astates[0]);sim.forward()
  anchors={c:_geom_id_image(sim,c) for c in CAMS}; anchor_renders={c:np.asarray(sim.render(width=256,height=256,camera_name=c))[::-1].copy() for c in CAMS}
  sim.set_state_from_flattened(states[a.frame]);sim.forward();curr={c:_geom_id_image(sim,c) for c in CAMS}
  if not return_arrays:a.output.mkdir(parents=True,exist_ok=True)
  files={};alignment={};arrays={}
  for c in CAMS:
   im,p=rgb(ds,a.episode,c,a.frame);aim,pp=rgb(ds,anchor,c,0)
   if im.shape!=(256,256,3) or aim.shape!=im.shape:raise ValueError('Unexpected image dimensions')
   render=np.asarray(sim.render(width=256,height=256,camera_name=c))[::-1]
   error=float(np.mean(np.abs(render.astype(np.float32)-im.astype(np.float32)))/255.)
   anchor_error=float(np.mean(np.abs(anchor_renders[c].astype(np.float32)-aim.astype(np.float32)))/255.)
   if error>.10 or anchor_error>.10:raise ValueError(f'Segmentation versus recorded RGB alignment is uncertain ({c}, current={error:.3f}, anchor={anchor_error:.3f})')
   alignment[c]={'current_mean_abs_rgb':error,'anchor_mean_abs_rgb':anchor_error}
   masks=np.stack([~np.isin(curr[c],excluded),~np.isin(anchors[c],excluded)])
   arrays[c]=dict(current=im,anchor=aim,current_background=masks[0],anchor_background=masks[1])
   if not return_arrays:
    cv2.imwrite(str(a.output/f'{c}_current.png'),cv2.cvtColor(im,cv2.COLOR_RGB2BGR));cv2.imwrite(str(a.output/f'{c}_anchor.png'),cv2.cvtColor(aim,cv2.COLOR_RGB2BGR));np.savez_compressed(a.output/f'{c}_masks.npz',current_background=masks[0],anchor_background=masks[1])
   files[str(p)]=sha(p);files[str(pp)]=sha(pp)
  all_ee,all_valid,ee_meta=episode_teacher(a.dataset,a.task,a.episode,anchor,position_scale_m=a.position_scale_m,rotation_scale_rad=a.rotation_scale_rad)
  if len(all_ee)!=len(states):raise ValueError('LeRobot State and MuJoCo Mask frame counts differ')
  ee,ee_valid=all_ee[a.frame],all_valid[a.frame]
  for p in [ep/'states.npz',ae/'states.npz',ep/'model.xml.gz',ae/'model.xml.gz',ep/'ep_meta.json',ae/'ep_meta.json',ep/'provenance.json',ae/'provenance.json',a.dataset/'human_anchor_pairs.csv']:files[str(p)]=sha(p)
  files.update(ee_meta['source_sha256'])
  if not return_arrays:np.savez_compressed(a.output/'geometry.npz',ee_latent=ee,ee_valid=ee_valid,ee_source=np.asarray('LeRobot observation.state'),ee_cache_version=np.asarray(EE_VERSION))
  m={'task':a.task,'episode':a.episode,'frame':a.frame,'anchor_episode':anchor,'anchor_generation_method':anchor_method,'reference_frame':0,'teacher_scope':'window_t_to_paired_anchor_frame_0','ee_site':'gripper0_right_grip_site','root_site':'robot0_right_center','ee_delta':'anchor minus current; rotation log(R_anchor_root @ R_current_root.T)','ee_teacher_source':'LeRobot observation.state','ee_cache_version':EE_VERSION,'ee_cache_file':EE_FILE,'ee_rotation_rule':'fixed hand-to-TCP right rotation cancels in Anchor times Current transpose','position_scale_m':a.position_scale_m,'rotation_scale_rad':a.rotation_scale_rad,'camera_order':list(CAMS),'dataset':str(a.dataset.resolve()),'file_sha256':files,'dynamic_geom_count':len(dynamic),'robot_geom_count':len(rg),'excluded_geom_count':len(excluded),'mask_policy':('robot_only_current_and_anchor_v2' if a.mask_policy=='robot_only' else 'robot_and_all_episode_moving_geoms_and_obj_names_current_and_anchor_v1'),'unassigned_segmentation_pixels':'retained unless rejected by RGB matching validity','rgb_alignment_mean_abs_threshold':.10,'rgb_alignment':alignment}
  if return_arrays:return dict(cameras=arrays,ee_latent=ee,ee_valid=ee_valid,metadata=m)
  (a.output/'metadata.json').write_text(json.dumps(m,indent=2));print(json.dumps(m,indent=2))
 finally:env.close()
if __name__=='__main__':main()
