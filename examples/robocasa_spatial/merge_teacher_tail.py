"""Merge newly generated tail-frame teachers with the existing full-chunk cache."""
import argparse, csv, json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--episodes-csv',type=Path,required=True)
p.add_argument('--old-root',type=Path,required=True)
p.add_argument('--tail-root',type=Path,required=True)
p.add_argument('--output-root',type=Path,required=True)
p.add_argument('--summary',type=Path,required=True)
a=p.parse_args()
summary={'episodes_expected':0,'episodes_merged':0,'episodes_missing':[],'windows_expected':0,'windows_merged':0,'valid_windows':0,'valid_dimensions':0}
with a.episodes_csv.open(newline='') as f: episodes=list(csv.DictReader(f))
for r in episodes:
 task,ep,n=r['task'],int(r['episode_index']),int(r['frames'])
 summary['episodes_expected']+=1; summary['windows_expected']+=n
 oldp=a.old_root/task/f'episode_{ep:06d}.npz'; tailp=a.tail_root/task/f'episode_{ep:06d}.npz'
 try:
  with np.load(oldp,allow_pickle=False) as z: old={k:z[k].copy() for k in z.files}
  with np.load(tailp,allow_pickle=False) as z: tail={k:z[k].copy() for k in z.files}
  om=json.loads(str(old['metadata'])); tm=json.loads(str(tail['metadata']))
  if om.get('task')!=task or tm.get('task')!=task or int(om.get('episode',-1))!=ep or int(tm.get('episode',-1))!=ep: raise ValueError('provenance mismatch')
  if om.get('teacher_latent_dim')!=64 or tm.get('teacher_latent_dim')!=64: raise ValueError('latent dim mismatch')
  oi=np.asarray(old['frame_index'],np.int32); ti=np.asarray(tail['frame_index'],np.int32)
  if set(map(int,oi)) & set(map(int,ti)): raise ValueError('old/tail frame overlap')
  frames=np.concatenate([oi,ti]); order=np.argsort(frames); frames=frames[order]
  if not np.array_equal(frames,np.arange(n,dtype=np.int32)): raise ValueError(f'frame coverage {len(frames)} != {n}')
  merged={}
  for key in old:
   if key=='metadata': continue
   if key not in tail: raise ValueError(f'missing tail field {key}')
   merged[key]=np.concatenate([old[key],tail[key]],axis=0)[order]
  meta=dict(tm); meta.update({'cache_version':'robocasa-frame-teacher-v4-exact-pose-reprojected-all-starts',
      'teacher_latent_dim':64,'frame_count':n,'merged_from':[str(oldp),str(tailp)],
      'frame_coverage':'all episode frames 0..T-1; official loader tail starts included'})
  merged['metadata']=np.asarray(json.dumps(meta,sort_keys=True))
  dst=a.output_root/task/f'episode_{ep:06d}.npz'; dst.parent.mkdir(parents=True,exist_ok=True)
  tmp=dst.with_suffix('.tmp.npz'); np.savez_compressed(tmp,**merged); tmp.replace(dst)
  valid=np.asarray(merged['teacher_valid'],bool)
  summary['episodes_merged']+=1; summary['windows_merged']+=len(frames)
  summary['valid_windows']+=int(valid.any(axis=1).sum()); summary['valid_dimensions']+=int(valid.sum())
 except Exception as e:
  summary['episodes_missing'].append({'task':task,'episode_index':ep,'expected_frames':n,'error':f'{type(e).__name__}: {e}'})
a.summary.parent.mkdir(parents=True,exist_ok=True)
a.summary.write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
if summary['episodes_merged'] != summary['episodes_expected']: raise SystemExit(2)
