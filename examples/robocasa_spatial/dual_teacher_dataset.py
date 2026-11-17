"""Native Groot sampling with exact episode/frame-aligned dual-teacher NPZ."""
import json
import hashlib
import os
from pathlib import Path
import numpy as np
from openpi.groot_utils.groot_openpi_dataset import GrootOpenpiMultiDataset

class OfficialMixtureWithDualTeacher(GrootOpenpiMultiDataset):
    def __init__(self, dataset_meta_list, action_horizon, teacher_root, dataset_weights=None):
        super().__init__(dataset_meta_list=dataset_meta_list, action_horizon=action_horizon,
                         dataset_weights=dataset_weights, dataset_weights_alpha=.4)
        self.root=Path(teacher_root)
        self.partial_smoke=os.environ.get('OPENPI_DUAL_TEACHER_PARTIAL_SMOKE')=='1'
        manifest=self.root/'manifest.json'
        if not self.partial_smoke:
            if not manifest.is_file():raise ValueError(f'Dual teacher cache is not finalized: {manifest}')
            audit=json.loads(manifest.read_text())
            if audit.get('status')!='complete' or audit.get('cache_version')!='roma_ee_window_v2_current_to_anchor':
                raise ValueError(f'Dual teacher cache manifest is not complete: {manifest}')
            if any(Path(meta['path']).resolve()!= (Path(audit['dataset'])/meta['task']/'lerobot').resolve() for meta in dataset_meta_list):
                raise ValueError('Dual teacher cache manifest points at a different dataset')
        self.tasks={str(Path(m['path']).resolve()):m['task'] for m in dataset_meta_list}
        self.cache={}; self._last_sample=None
    def sample_step(self,index):
        self._last_sample=super().sample_step(index)
        return self._last_sample
    def __getitem__(self,index):
        item=super().__getitem__(index)
        dataset,episode,frame=self._last_sample
        task=self.tasks[str(Path(dataset.dataset_path).resolve())];episode=int(episode);frame=int(frame)
        key=(task,episode)
        if key not in self.cache:
            path=self.root/task/f'episode_{episode:06d}.npz'
            if not path.exists():
                if not self.partial_smoke:raise FileNotFoundError(f'Missing finalized teacher episode: {path}')
                self.cache[key]=None
            else:
                with np.load(path,allow_pickle=False) as z:
                    meta=json.loads(str(z['metadata']))
                    if meta.get('cache_version')!='roma_ee_window_v2_current_to_anchor' or meta.get('task')!=task or int(meta.get('episode',-1))!=episode:
                        raise ValueError(f'Wrong dual teacher provenance: {path}')
                    from examples.robocasa_dual_teacher.provenance import verify_sources
                    verify_sources(meta,task,episode,dataset.dataset_path)
                    frames=np.asarray(z['frame_index'],np.int64)
                    scene=np.asarray(z['scene_latent'],np.float32);sm=np.asarray(z['scene_valid'],bool)
                    ee=np.asarray(z['ee_latent'],np.float32);em=np.asarray(z['ee_valid'],bool)
                    if len(np.unique(frames))!=len(frames) or scene.shape!=(len(frames),64) or sm.shape!=scene.shape or ee.shape!=(len(frames),6) or em.shape!=ee.shape:
                        raise ValueError(f'Invalid dual teacher shapes: {path}')
                    if not np.isfinite(scene[sm]).all() or not np.isfinite(ee[em]).all():
                        raise ValueError(f'Non-finite valid teacher: {path}')
                    self.cache[key]=({int(f):i for i,f in enumerate(frames)},scene.copy(),sm.copy(),ee.copy(),em.copy())
        cached=self.cache[key];row=None if cached is None else cached[0].get(frame)
        if row is None:
            if not self.partial_smoke:raise ValueError(f'Missing finalized teacher frame: {task} EP{episode} frame{frame}')
            item['teacher_latent']=np.zeros(70,np.float32);item['teacher_valid']=np.zeros(70,bool)
        else:
            item['teacher_latent']=np.concatenate([cached[1][row],cached[3][row]])
            item['teacher_valid']=np.concatenate([cached[2][row],cached[4][row]])
        item.update(task=task,episode_index=episode,frame_index=frame)
        return item
