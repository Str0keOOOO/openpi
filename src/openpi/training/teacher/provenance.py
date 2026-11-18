"""Verify teacher provenance against the recorded dataset and actual Anchor."""
import hashlib
from functools import lru_cache
from pathlib import Path

@lru_cache(maxsize=8192)
def source_episode(dataset_path, episode):
    """Follow the view's extras link to the dataset and episode used by teacher caches."""
    extras = (Path(dataset_path) / 'extras' / f'episode_{episode:06d}').resolve()
    return extras.parent.parent, int(extras.name.removeprefix('episode_'))

def file_digest(path):
 path=Path(path);s=path.stat()
 return _cached_digest(str(path.resolve()),s.st_size,s.st_mtime_ns,s.st_ctime_ns)

@lru_cache(maxsize=32768)
def _cached_digest(path,size,mtime,ctime):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def verify_sources(meta, task, episode, dataset_path):
 root=Path(meta['dataset']).resolve();ds=root/task/'lerobot'
 if ds.resolve()!=Path(dataset_path).resolve():raise ValueError('Teacher dataset root mismatch')
 if meta.get('task')!=task or int(meta.get('episode',-1))!=episode:raise ValueError('Teacher episode mismatch')
 if meta.get('reference_frame')!=0:raise ValueError('Teacher reference must be Anchor frame 0')
 anchor=int(meta['anchor_episode']);source=ds/'extras'/f'episode_{episode:06d}';reference=ds/'extras'/f'episode_{anchor:06d}'
 required={str(root/'human_anchor_pairs.csv')}
 for directory in [source,reference]:
  required.update(str(directory/n) for n in ['states.npz','model.xml.gz','ep_meta.json','provenance.json'])
 for cam in ('robot0_agentview_left','robot0_agentview_right'):
  for ep in [episode,anchor]:required.add(str(ds/'videos'/f'chunk-{ep//1000:03d}'/f'observation.images.{cam}'/f'episode_{ep:06d}.mp4'))
 signatures=meta.get('file_sha256')
 if not isinstance(signatures,dict) or not required.issubset(signatures):raise ValueError('Teacher provenance does not cover every required source file')
 for path in required:
  if file_digest(path)!=signatures[path]:raise ValueError('Teacher source changed: '+path)
 return True
