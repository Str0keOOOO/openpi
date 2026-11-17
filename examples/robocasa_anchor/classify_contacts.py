"""Identify MuJoCo geom/body pairs recorded in an Anchor pilot."""
import argparse,json,collections
from pathlib import Path
import robocasa,robosuite
from robocasa.utils import lerobot_utils as LU
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--episode',type=int,required=True);p.add_argument('--pilot',type=Path,required=True);a=p.parse_args()
root=Path('/data4/xuchenfei/robocasa365/data/household_augmented')/a.task/'lerobot';m=LU.get_env_metadata(root);kw=dict(m['env_kwargs']);kw.update(env_name=m['env_name'],has_renderer=False,has_offscreen_renderer=False,use_camera_obs=False,seed=0,ignore_done=True)
e=robosuite.make(**kw)
try:
 reset_to(e,dict(model=LU.get_episode_model_xml(root,a.episode),states=LU.get_episode_states(root,a.episode)[0],ep_meta=json.dumps(LU.get_episode_meta(root,a.episode))))
 events=json.loads((a.pilot/'operation_result.json').read_text())['deep_contacts_unclassified'];pairs={}
 for x in events:
  ids=(x['geom1'],x['geom2']);key=tuple(sorted(ids))
  if key not in pairs:
   geoms=[]
   for i in key:
    body=int(e.sim.model.geom_bodyid[i]);geoms.append({'geom_id':i,'geom_name':e.sim.model.geom_id2name(i),'body_id':body,'body_name':e.sim.model.body_id2name(body)})
   pairs[key]={'geoms':geoms,'count':0,'min_distance_m':0.,'first_step':x['step'],'last_step':x['step']}
  o=pairs[key];o['count']+=1;o['min_distance_m']=min(o['min_distance_m'],x['distance_m']);o['last_step']=x['step']
 result={'event_count':len(events),'pairs':list(pairs.values())}
 (a.pilot/'contact_pairs.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
finally:e.close()
