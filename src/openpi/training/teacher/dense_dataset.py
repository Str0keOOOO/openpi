"""Native Groot sampling plus per-window dense RoMa fields for online augmentation."""
from pathlib import Path
from collections import OrderedDict
import os,json
import numpy as np
from openpi.groot_utils.groot_openpi_dataset import GrootOpenpiMultiDataset
from openpi.training.teacher.provenance import verify_sources
from openpi.training.teacher.schema import VERSION,ROMA_SETTING,BIDIRECTIONAL,validate_metadata
from openpi.training.teacher.state_ee import EE_VERSION,load_teacher
class DenseSpatialInputs:
 def __init__(self,action_dim):self.action_dim=action_dim
 def __call__(self,data):
  from openpi.policies.robocasa_policy import RobocasaInputs
  from openpi.models.model import ModelType
  out=RobocasaInputs(self.action_dim,ModelType.PI05)(data)
  keys=['scene_flow','scene_pixel_valid','ee_teacher','ee_teacher_valid']
  if any(k in data for k in keys):
   if not all(k in data for k in keys):raise ValueError('Partial dense teacher batch')
   for k in keys:out[k]=data[k]
  return out
class OfficialMixtureWithDenseTeacher(GrootOpenpiMultiDataset):
 def __init__(self,dataset_meta_list,action_horizon,teacher_root,dataset_weights=None):
  super().__init__(dataset_meta_list=dataset_meta_list,action_horizon=action_horizon,dataset_weights=dataset_weights,dataset_weights_alpha=.4)
  self.root=Path(teacher_root);self.tasks={str(Path(x['path']).resolve()):x['task'] for x in dataset_meta_list};self.meta=OrderedDict();self._last_sample=None
  if os.environ.get('OPENPI_DENSE_TEACHER_SMOKE')!='1':
   m=json.loads((self.root/'manifest.json').read_text())
   if m.get('status')!='complete' or m.get('cache_version')!=VERSION:raise ValueError('Dense teacher cache not complete')
   if m['teacher_settings'].get('roma_setting')!=ROMA_SETTING or m['teacher_settings'].get('roma_bidirectional') is not BIDIRECTIONAL:raise ValueError('Training requires the official Fast single-direction teacher')
   if m['teacher_settings'].get('ee_cache_version')!=EE_VERSION:raise ValueError('Training requires EE labels from recorded State')
 def sample_step(self,index):self._last_sample=super().sample_step(index);return self._last_sample
 def __getitem__(self,index):
  item=super().__getitem__(index);ds,ep,t=self._last_sample;ep=int(ep);t=int(t);task=self.tasks[str(Path(ds.dataset_path).resolve())];key=(task,ep);folder=self.root/task/f'episode_{ep:06d}'
  if key not in self.meta:
   m=json.loads((folder/'metadata.json').read_text())
   validate_metadata(m)
   if (Path(ds.dataset_path)/'data').resolve()!=(Path(m['dataset'])/task/'lerobot'/'data').resolve():raise ValueError('Training view and dense teacher source differ')
   verify_sources(m,task,ep,Path(m['dataset'])/task/'lerobot')
   self.meta[key]=m
   if len(self.meta)>32:self.meta.popitem(last=False)
  else:self.meta.move_to_end(key)
  with np.load(folder/f'frame_{t:06d}.npz',allow_pickle=False) as z:
   if int(z['frame_index'])!=t or str(z['cache_version'])!=VERSION or str(z['roma_setting'])!=ROMA_SETTING:raise ValueError('Incorrect teacher frame, version or matching mode')
   flow=np.asarray(z['scene_flow'],np.float32);valid=np.unpackbits(z['scene_valid_bits'],axis=-1,count=256*256).reshape(2,256,256).astype(bool)
  all_ee,all_ev,ee_meta=load_teacher(folder)
  if ee_meta['task']!=task or ee_meta['episode']!=ep or ee_meta['anchor_episode']!=self.meta[key]['anchor_episode']:raise ValueError('State EE cache provenance mismatch')
  ee,ev=all_ee[t],all_ev[t]
  if flow.shape!=(2,256,256,2) or ee.shape!=(6,) or ev.shape!=(6,):raise ValueError('Invalid dense teacher shape')
  if not np.isfinite(flow[valid]).all() or not np.isfinite(ee[ev]).all():raise ValueError('Nonfinite valid teacher')
  item.update(scene_flow=flow,scene_pixel_valid=valid,ee_teacher=ee,ee_teacher_valid=ev)
  return item
