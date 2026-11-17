"""Probe an episode anchor by rebuilding its scene; output is diagnostic, not verified labels."""
import argparse, gzip, json, pathlib, xml.etree.ElementTree as ET
import numpy as np
import robosuite as suite
import robocasa  # registers environments
from robocasa.utils import env_utils

p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--episode-dir',type=pathlib.Path,required=True);p.add_argument('--seed',type=int,default=0);a=p.parse_args()
m=json.loads((a.episode_dir/'ep_meta.json').read_text());root=ET.fromstring(gzip.open(a.episode_dir/'model.xml.gz','rt').read())
bodies={b.get('name'):b for b in root.findall('./worldbody/body')}
e=suite.make(a.task,robots='PandaOmron',has_renderer=False,has_offscreen_renderer=False,use_camera_obs=False,layout_and_style_ids=[(m['layout_id'],m['style_id'])],seed=a.seed)
try:
 e.set_ep_meta(m);e.reset(); changed=[]; matched=[]
 for name,f in e.fixtures.items():
  b=bodies.get(name+'_main')
  if b is None or not hasattr(f,'_obj'): continue
  old=f._obj.get('pos'); pos=b.get('pos')
  if pos is None: continue
  matched.append(name)
  if old!=pos: changed.append(name)
  for attr in ('pos','quat','euler'):
   if attr in b.attrib:f._obj.set(attr,b.get(attr))
   elif attr in f._obj.attrib: del f._obj.attrib[attr]
 if a.task=='OpenDrawer':
  text=m.get('lang','').lower();drawer=e.drawer
  if 'left drawer' in text: offset=(drawer.width/2+0.20,-0.10)
  elif 'right drawer' in text: offset=(-drawer.width/2-0.20,-0.10)
  else: raise ValueError('drawer side is missing or ambiguous in episode instruction')
  pos,ori=env_utils.compute_robot_base_placement_pose(e,ref_fixture=drawer,offset=offset)
 else: pos,ori=env_utils.init_robot_base_pose(e)
 yaw_error = float((float(ori[2])-float(m['init_robot_base_ori'][2])+np.pi)%(2*np.pi)-np.pi)
 status = 'rejected_yaw_mismatch' if abs(yaw_error)>np.deg2rad(0.1) else 'candidate_needs_validation'
 print('ANCHOR_PROBE_JSON',json.dumps({'yaw_error_deg':float(np.degrees(yaw_error)),'task':a.task,'episode_dir':str(a.episode_dir),'seed':a.seed,'anchor_xyz':np.asarray(pos).tolist(),'anchor_rpy':np.asarray(ori).tolist(),'human_xyz':m['init_robot_base_pos'],'human_rpy':m['init_robot_base_ori'],'xml_fixture_bodies_matched':len(matched),'xml_fixture_bodies_changed':changed,'status':status}),flush=True)
finally:e.close()
