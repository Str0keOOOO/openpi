"""Numerical geometry/masking and cache provenance smoke checks."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from examples.robocasa_dual_teacher.geometry import ee_difference,pool_match
from examples.robocasa_dual_teacher.provenance import verify_sources

def T(x,rot):
 z=np.eye(4);z[:3,:3]=Rotation.from_rotvec(rot).as_matrix();z[:3,3]=x;return z

def main():
 root=T([.1,.2,.3],[0,0,.2]);eroot=T([.3,.1,.8],[0,.1,0]);aroot=T([-.1,.3,.4],[0,0,-.1]);ae=T([.4,.2,.6],[.2,0,0]);world=T([3,-2,1],[.1,.2,-.3])
 ref=ee_difference(root,eroot,aroot,ae,position_scale_m=.1,rotation_scale_rad=1)
 shifted=ee_difference(world@root,world@eroot,world@aroot,world@ae,position_scale_m=.1,rotation_scale_rad=1)
 assert np.allclose(ref,shifted,atol=1e-6)
 assert np.allclose(ee_difference(root,eroot,root,eroot,position_scale_m=.1,rotation_scale_rad=1),0,atol=1e-6)
 h=w=256;flow=np.zeros((h,w,2),np.float32);flow[...,0]=4;flow[...,1]=-2;back=-flow;conf=np.ones((h,w),np.float32);mask=np.ones((h,w),bool)
 pooled,valid,coverage,pixels=pool_match(flow,mask,mask,back,conf,min_valid_fraction=.1)
 assert valid.all() and np.allclose(pooled.reshape(-1,2),[4/w,-2/h],atol=1e-6)
 zero=np.zeros_like(mask);pooled,valid,coverage,pixels=pool_match(flow,zero,mask,back,conf,min_valid_fraction=.1);assert not valid.any() and not pixels.any()
 root=Path('/data4/xuchenfei/robocasa365/output/roma_ee_dual_teacher/smoke');cache=root/'cache_v1/PickPlaceSinkToCounter/episode_000086.npz'
 with np.load(cache,allow_pickle=False) as z:
  m=json.loads(str(z['metadata']));verify_sources(m,'PickPlaceSinkToCounter',86,Path(m['dataset'])/'PickPlaceSinkToCounter/lerobot')
  assert z['frame_index'].tolist()==[0] and z['scene_latent'].shape==(1,64) and z['ee_latent'].shape==(1,6)
  assert int(z['scene_valid'].sum())>0 and z['ee_valid'].all()
  geometry=np.load(root/'capture_v1/geometry.npz');expected=ee_difference(geometry['current_root'],geometry['current_ee'],geometry['anchor_root'],geometry['anchor_ee'],position_scale_m=m['position_scale_m'],rotation_scale_rad=m['rotation_scale_rad'])
  # Historical v1 smoke fixtures intentionally retain the old direction.
  if m.get('cache_version') == 'roma_ee_window_v1':expected=-expected
  elif m.get('cache_version') != 'roma_ee_window_v2_current_to_anchor':raise ValueError('Unknown teacher direction version')
  assert np.allclose(z['ee_latent'][0],expected,atol=1e-6)
 result=dict(status='passed',synthetic=['global_rigid_transform_invariance','identical_pose_zero','known_flow_direction_and_units','empty_mask'],real=['source_hash_and_anchor_provenance','scene_and_ee_shapes_masks','ee_fk_transform_recompute'])
 (root/'geometry_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
