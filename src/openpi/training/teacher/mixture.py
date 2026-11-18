"""Official Groot reads and samples all observations; append matching teacher fields only."""
import json
from pathlib import Path
import numpy as np
from openpi.groot_utils.groot_openpi_dataset import GrootOpenpiMultiDataset
from openpi.training.teacher.provenance import source_episode


class OfficialMixtureWithTeacher(GrootOpenpiMultiDataset):
    def __init__(self, dataset_meta_list, action_horizon, teacher_root, dataset_weights=None):
        super().__init__(dataset_meta_list=dataset_meta_list, action_horizon=action_horizon,
                         dataset_weights=dataset_weights, dataset_weights_alpha=0.4)
        self.teacher_root=Path(teacher_root)
        self._task_by_path={str(Path(meta['path']).resolve()):meta['task'] for meta in dataset_meta_list}
        self._teachers={}
        self._last_sample=None

    def sample_step(self,index):
        # Same task/episode/frame sampler as the native project, with provenance retained.
        sample=super().sample_step(index)
        self._last_sample=sample
        return sample

    def __getitem__(self,index):
        # Native Groot handles three RGB streams, state/action ordering, language,
        # all frame starts, and action padding at episode boundaries.
        item=super().__getitem__(index)
        dataset,episode,frame=self._last_sample
        task=self._task_by_path[str(Path(dataset.dataset_path).resolve())]
        _, episode = source_episode(dataset.dataset_path, int(episode))
        frame = int(frame)
        key=(task,episode)
        if key not in self._teachers:
            path=self.teacher_root/task/f'episode_{episode:06d}.npz'
            if path.is_file():
                with np.load(path,allow_pickle=False) as z:
                    metadata=json.loads(str(z['metadata']))
                    if metadata.get('task')!=task or int(metadata.get('episode',-1))!=episode:
                        raise ValueError(f'teacher provenance mismatch: {path}')
                    if metadata.get('cache_version') not in (
                        'robocasa-frame-teacher-v4-exact-pose-reprojected',
                        'robocasa-frame-teacher-v4-exact-pose-reprojected-all-starts',
                    ):
                        raise ValueError(
                            f'outdated teacher cache at {path}: expected exact-pose reprojection '
                            "(robocasa-frame-teacher-v4-exact-pose-reprojected); rebuild the cache before training"
                        )
                    frames=np.asarray(z['frame_index'],np.int64)
                    latent=np.asarray(z['teacher_latent'],np.float32).copy()
                    valid=np.asarray(z['teacher_valid'],bool).copy()
                    if latent.shape!=(len(frames),64) or valid.shape!=latent.shape:
                        raise ValueError(f'invalid 64D teacher shape: {path}')
                    self._teachers[key]=({int(f):i for i,f in enumerate(frames)},latent,valid)
            else:self._teachers[key]=None
        cached=self._teachers[key]
        row=None if cached is None else cached[0].get(frame)
        if row is None:
            # New native end-of-episode starts have no cached teacher yet.
            # No label is substituted from another frame; Human actions are retained.
            item['teacher_latent']=np.zeros(64,np.float32)
            item['teacher_valid']=np.zeros(64,bool)
        else:
            item['teacher_latent']=cached[1][row]
            item['teacher_valid']=cached[2][row]
        item['task']=task;item['episode_index']=episode;item['frame_index']=frame
        return item
