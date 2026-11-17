"""Anchor poses require per-episode provenance; no global or mean-pose fallback."""
import json
from pathlib import Path
import numpy as np

class AnchorProvider:
    def __init__(self, directory): self.directory=Path(directory)
    def get(self, task, episode):
        p=self.directory/task/f'episode_{episode:06d}.json'
        if not p.is_file(): raise FileNotFoundError(f'No verified reset anchor: {p}; provide per-episode fixture-derived pose and provenance')
        d=json.loads(p.read_text())
        if not d.get('source') or not d.get('world_from_anchor'):
            raise ValueError(f'Anchor must contain source and world_from_anchor: {p}')
        t=np.asarray(d['world_from_anchor'],dtype=float)
        if t.shape != (4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1]): raise ValueError(f'invalid anchor matrix: {p}')
        r=t[:3,:3]
        if not np.allclose(r.T@r,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(r),1,atol=1e-5): raise ValueError(f'invalid rotation: {p}')
        return t,d
