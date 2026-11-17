"""Validate complete per-frame teacher coverage before allowing training."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pyarrow.parquet as pq
from examples.robocasa_dual_teacher.provenance import verify_sources

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
 rows=[];errors=[];settings=None;checkpoint_hashes={}
 for prov in sorted(a.dataset.glob('*/lerobot/extras/episode_*/provenance.json')):
  ds=prov.parents[2];task=prov.parents[3].name;ep=int(prov.parent.name.split('_')[-1]);data=ds/'data'/f'chunk-{ep//1000:03d}'/f'episode_{ep:06d}.parquet';cache=a.cache/task/f'episode_{ep:06d}.npz'
  n=pq.read_metadata(data).num_rows
  try:
   with np.load(cache,allow_pickle=False) as z:
    m=json.loads(str(z['metadata']));f=np.asarray(z['frame_index']);sv=np.asarray(z['scene_valid'],bool);ev=np.asarray(z['ee_valid'],bool)
    if m.get('cache_version')!='roma_ee_window_v2_current_to_anchor' or Path(m['dataset']).resolve()!=a.dataset.resolve():raise ValueError('wrong dataset or version')
    if not np.array_equal(f,np.arange(n)) or sv.shape!=(n,64) or ev.shape!=(n,6):raise ValueError('incomplete frame coverage or wrong shape')
    verify_sources(m,task,ep,ds)
    sig={k:m[k] for k in ('position_scale_m','rotation_scale_rad','ee_delta','scene_direction','pooling','roma_checkpoint','roma_checkpoint_sha256','roma_seed','mask_policy','confidence_threshold','fb_threshold_px','min_valid_fraction')}
    if settings is None:settings=sig
    elif settings!=sig:raise ValueError('teacher model or normalization settings differ across episodes')
    checkpoint=Path(m['roma_checkpoint']);key=(str(checkpoint),checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns)
    if key not in checkpoint_hashes:
     h=hashlib.sha256()
     with checkpoint.open('rb') as q:
      for b in iter(lambda:q.read(1024*1024),b''):h.update(b)
     checkpoint_hashes[key]=h.hexdigest()
    if checkpoint_hashes[key]!=m['roma_checkpoint_sha256']:raise ValueError('RoMa checkpoint differs')
    if not ev.all():raise ValueError('EE labels missing')
    if not np.isfinite(z['scene_latent'][sv]).all() or not np.isfinite(z['ee_latent'][ev]).all():raise ValueError('non-finite valid labels')
    rows.append(dict(task=task,episode=ep,frames=n,scene_valid_dims=int(sv.sum()),ee_valid_dims=int(ev.sum()),scene_windows_without_valid=int((~sv.any(axis=1)).sum())))
  except Exception as e:errors.append(dict(task=task,episode=ep,error=str(e)))
 report=dict(dataset=str(a.dataset.resolve()),episodes_ok=len(rows),episodes_failed=len(errors),frames=sum(x['frames'] for x in rows),episodes=rows,errors=errors)
 a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('episodes','errors')}))
 manifest=a.cache/'manifest.json'
 if errors:
  # Never leave an old approval in place after a failed re-audit.
  manifest.unlink(missing_ok=True);raise SystemExit('Cache incomplete or stale; see report')
 tmp=manifest.with_suffix('.tmp');tmp.write_text(json.dumps(dict(status='complete',dataset=report['dataset'],episodes=report['episodes_ok'],frames=report['frames'],cache_version='roma_ee_window_v2_current_to_anchor',teacher_settings=settings),indent=2));tmp.replace(manifest)
if __name__=='__main__':main()
