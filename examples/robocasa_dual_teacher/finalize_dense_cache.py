"""Full precision/shape/provenance audit; never approve incomplete dense caches."""
import argparse,json,time,concurrent.futures
from pathlib import Path
import numpy as np
from examples.robocasa_dual_teacher.provenance import verify_sources,file_digest
from examples.robocasa_dual_teacher.dense_worker import VERSION,CAMS,write
from examples.robocasa_dual_teacher.cache_schema import validate_metadata
from examples.robocasa_dual_teacher.state_ee import load_teacher,EE_VERSION

def audit(job,root,dataset):
 task=job['task'];ep=int(job['episode']);n=int(job['frames']);folder=root/task/f'episode_{ep:06d}'
 try:
  completion=json.loads((folder/'complete.json').read_text());m=json.loads((folder/'metadata.json').read_text())
  validate_metadata(m)
  if m.get('correlation_backends_used')!=['official_native_pytorch']:raise ValueError('Non-official inference backend')
  if completion.get('frames')!=n or m.get('source_frames')!=n or m.get('cache_version')!=VERSION:raise ValueError('incomplete episode/version')
  verify_sources(m,task,ep,dataset/task/'lerobot')
  if m['camera_order']!=list(CAMS) or m['reference_frame']!=0 or m['position_scale_m']!=.1 or m['rotation_scale_rad']!=1.:raise ValueError('wrong camera/reference/EE scales')
  if file_digest(m['roma_checkpoint'])!=m['roma_checkpoint_sha256']:raise ValueError('RoMa checkpoint changed')
  all_ee,all_ev,ee_meta=load_teacher(folder)
  if all_ee.shape!=(n,6) or ee_meta['task']!=task or ee_meta['episode']!=ep or ee_meta['anchor_episode']!=m['anchor_episode']:raise ValueError('Recorded State EE provenance/shape mismatch')
  if m.get('ee_cache_version')!=EE_VERSION:raise ValueError('Missing State EE cache metadata')
  scene_pixels=np.zeros(2,np.int64);valid_cells=0;no_scene=0;size=0
  for t in range(n):
   path=folder/f'frame_{t:06d}.npz'
   with np.load(path,allow_pickle=False) as z:
    f=z['scene_flow'];bits=z['scene_valid_bits'];ee=all_ee[t];ev=all_ev[t];rat=z['valid_pixel_ratio']
    if int(z['frame_index'])!=t or f.shape!=(2,256,256,2) or bits.shape!=(2,8192) or ee.shape!=(6,) or ev.shape!=(6,) or rat.shape!=(2,) or str(z['cache_version'])!=VERSION or str(z['roma_setting'])!=m['roma_setting']:raise ValueError(f'frame {t}: shape, index, version or mode')
    mask=np.unpackbits(bits,axis=-1,count=65536).reshape(2,256,256).astype(bool)
    if not np.isfinite(f[mask]).all() or not np.isfinite(ee[ev]).all() or not ev.all():raise ValueError(f'frame {t}: invalid real labels')
    if 'correlation_backend' in z and int(z['correlation_backend'])==1:
     proof=Path(m['fused_validation_report'])
     if file_digest(proof)!=m.get('fused_validation_sha256') or json.loads(proof.read_text()).get('status')!='passed':raise ValueError('Fused-kernel equivalence proof changed or missing')
    ratios=mask.mean(axis=(1,2))
    if not np.allclose(rat,ratios,atol=1e-6):raise ValueError(f'frame {t}: pixel mask ratio inconsistent')
    counts=mask.reshape(2,4,64,4,64).sum(axis=(2,4));valid_cells+=int((counts>=.1*4096).sum());no_scene+=int(not (counts>=.1*4096).any());scene_pixels+=mask.sum(axis=(1,2));size+=path.stat().st_size
  settings={k:m[k] for k in ['roma_checkpoint_sha256','roma_seed','roma_setting','roma_bidirectional','mask_policy','scene_direction','confidence_threshold','fb_threshold_px','min_valid_fraction','ee_delta']}
  settings.update(ee_cache_version=EE_VERSION,ee_teacher_source='LeRobot observation.state')
  return dict(task=task,episode=ep,frames=n,status='passed',valid_background_pixels=scene_pixels.tolist(),unaugmented_valid_grid_count=valid_cells,windows_without_valid_scene=no_scene,bytes=size,settings=settings)
 except Exception as e:return dict(task=task,episode=ep,frames=n,status='failed',error=str(e))

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--jobs',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--audit-only',action='store_true');a=p.parse_args();jobs=json.loads(a.jobs.read_text());start=time.time();rows=[]
 if not a.audit_only:
  import pyarrow.parquet as pq
  expected=set()
  for path in a.dataset.glob('*/lerobot/data/chunk-*/episode_*.parquet'):
   expected.add((path.parents[3].name,int(path.stem.split('_')[-1]),pq.read_metadata(path).num_rows))
  actual={(j['task'],int(j['episode']),int(j['frames'])) for j in jobs}
  if expected!=actual or len(actual)!=len(jobs):raise ValueError('Jobs do not cover the exact final dataset')
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  futures=[pool.submit(audit,j,a.cache,a.dataset) for j in jobs]
  for future in concurrent.futures.as_completed(futures):
   rows.append(future.result())
   if len(rows)%25==0:print(json.dumps(dict(audited=len(rows),total=len(jobs),failed=sum(r['status']=='failed' for r in rows))),flush=True)
 failed=[r for r in rows if r['status']!='passed'];good=[r for r in rows if r['status']=='passed'];settings=good[0]['settings'] if good else None
 if any(r['settings']!=settings for r in good):failed.append(dict(error='inconsistent settings'))
 report=dict(status='failed' if failed else 'passed',cache_version=VERSION,episodes=len(rows),frames=sum(r['frames'] for r in good),errors=failed,episodes_results=rows,elapsed_seconds=time.time()-start)
 write(a.report,report)
 if failed:
  if not a.audit_only:(a.cache/'manifest.json').unlink(missing_ok=True)
  raise SystemExit('Dense audit failed; see report')
 if a.audit_only:return
 write(a.cache/'manifest.json',dict(status='complete',cache_version=VERSION,dataset=str(a.dataset.resolve()),episodes=len(rows),frames=report['frames'],teacher_settings=settings,audit_report=str(a.report),created_at=time.time()))
if __name__=='__main__':main()
