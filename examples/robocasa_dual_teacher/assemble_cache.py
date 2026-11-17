"""Assemble independently generated windows into a native episode teacher cache."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
def assemble(paths,root):
 windows=[];base=None;digests={}
 for path in paths:
  with np.load(path,allow_pickle=False) as z:
   m=json.loads(str(z['metadata']));f=np.asarray(z['frame_index']);s=np.asarray(z['scene_latent']);sm=np.asarray(z['scene_valid']);e=np.asarray(z['ee_latent']);em=np.asarray(z['ee_valid'])
   if m.get('cache_version')!='roma_ee_window_v2_current_to_anchor' or f.shape!=(1,) or int(f[0])!=m['frame'] or s.shape!=(1,64) or sm.shape!=s.shape or e.shape!=(1,6) or em.shape!=e.shape:raise ValueError(f'Invalid window: {path}')
   for p,h in m['file_sha256'].items():
    source=Path(p)
    if not source.is_file():raise ValueError(f'Missing dataset source: {p}')
    stat=source.stat();key=(p,stat.st_size,stat.st_mtime_ns)
    if key not in digests:digests[key]=digest(p)
    if digests[key]!=h:raise ValueError(f'Stale dataset source: {p}')
   keys=('task','episode','anchor_episode','anchor_generation_method','reference_frame','dataset','position_scale_m','rotation_scale_rad','camera_order','ee_delta','scene_direction','pooling','roma_checkpoint','roma_checkpoint_sha256','roma_seed','mask_policy','file_sha256','confidence_threshold','fb_threshold_px','min_valid_fraction')
   signature={k:m[k] for k in keys}
   checkpoint=Path(m['roma_checkpoint']);stat=checkpoint.stat();key=(str(checkpoint),stat.st_size,stat.st_mtime_ns)
   if key not in digests:digests[key]=digest(checkpoint)
   if digests[key]!=m['roma_checkpoint_sha256']:raise ValueError('RoMa checkpoint checksum mismatch')
   if base is None:base=signature
   elif signature!=base:raise ValueError('Mixed source, anchor, model settings or dataset version')
   windows.append((int(f[0]),s[0].copy(),sm[0].copy(),e[0].copy(),em[0].copy()))
 if not windows:raise ValueError('No valid input windows')
 windows.sort(key=lambda x:x[0]);frames=[x[0] for x in windows]
 if len(set(frames))!=len(frames):raise ValueError('Duplicate window frame index')
 out=Path(root)/base['task']/f"episode_{base['episode']:06d}.npz";out.parent.mkdir(parents=True,exist_ok=True)
 tmp=out.with_name(out.stem+'.tmp.npz');meta=dict(base,cache_version='roma_ee_window_v2_current_to_anchor',window_count=len(windows))
 np.savez_compressed(tmp,frame_index=frames,scene_latent=np.stack([x[1] for x in windows]),scene_valid=np.stack([x[2] for x in windows]),ee_latent=np.stack([x[3] for x in windows]),ee_valid=np.stack([x[4] for x in windows]),metadata=np.asarray(json.dumps(meta)))
 tmp.replace(out);return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--window',type=Path,nargs='+',required=True);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args();print(assemble(a.window,a.output_root))
if __name__=='__main__':main()
