"""Independent real teacher cache; missing supervision stays missing."""
import json
from pathlib import Path
import numpy as np

CAMERAS = ('robot0_agentview_left', 'robot0_agentview_right')
LATENT_DIM = len(CAMERAS) * 32

def save(path, latents, masks, metadata, *, pixel_robot_masks_current=None, pixel_robot_masks_reference=None, pixel_flow_masks=None):
    path = Path(path)
    if set(latents) != set(CAMERAS) or set(masks) != set(CAMERAS): raise ValueError('left/right agentview entries required')
    required = ('task', 'episode', 'frame', 'sharp_version', 'anchor_source', 'calibration_id', 'cache_version')
    if any(k not in metadata for k in required): raise ValueError('missing teacher provenance')
    z = np.concatenate([np.asarray(latents[c], dtype=np.float32).reshape(32) for c in CAMERAS])
    m = np.concatenate([np.asarray(masks[c], dtype=bool).reshape(32) for c in CAMERAS])
    if not np.isfinite(z[m]).all(): raise ValueError('non-finite valid latent')
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays={'latent':z,'valid':m,'metadata':json.dumps(metadata, sort_keys=True)}
    for label, source in [('robot_mask_current',pixel_robot_masks_current),('robot_mask_reference',pixel_robot_masks_reference),('flow_valid',pixel_flow_masks)]:
        if source is None: continue
        if set(source)!=set(CAMERAS): raise ValueError(f'{label} must contain exactly the two agentview cameras')
        for cam in CAMERAS:
            value=np.asarray(source[cam],dtype=bool)
            if value.ndim!=2: raise ValueError(f'{label}[{cam}] must be HxW')
            arrays[f'{label}_{cam}']=value
    np.savez_compressed(path, **arrays)

def load(path):
    with np.load(path, allow_pickle=False) as f:
        z,m = f['latent'].copy(),f['valid'].copy()
        meta=json.loads(str(f['metadata']))
    if z.shape != (LATENT_DIM,) or m.shape != (LATENT_DIM,): raise ValueError(f'invalid teacher cache shape, expected {LATENT_DIM}')
    return z,m,meta
