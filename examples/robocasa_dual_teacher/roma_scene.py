"""Offline RoMa v2 Current→Anchor teacher; geometry is used only for masks."""
import argparse,hashlib,json,sys,time,functools
from pathlib import Path
import cv2,numpy as np,torch
from examples.robocasa_dual_teacher.geometry import pool_match
CAMS=('robot0_agentview_left','robot0_agentview_right')
from examples.robocasa_dual_teacher.roma_backend import ROOT as TEACHER_ROOT,setup_roma

@functools.lru_cache(maxsize=4)
def checkpoint_sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def load_roma(device, *, setting='fast', bidirectional=None):
 torch.manual_seed(0)
 if device=='cuda':torch.cuda.manual_seed_all(0)
 if device=='cpu' and torch.cuda.is_available():raise RuntimeError('For CPU inference, set CUDA_VISIBLE_DEVICES= before starting the process')
 model,predict,info=setup_roma(setting=setting,bidirectional=bidirectional)
 if device=='cuda':
  if not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
  model.cuda()
 elif device!='cpu':raise ValueError('device must be cpu or cuda')
 return model,predict,info

def process(capture,output,predict,*,confidence_threshold=None,fb_threshold_px=None,min_valid_fraction=.1,model_info=None,save_debug=False):
 capture,output=Path(capture),Path(output);meta=json.loads((capture/'metadata.json').read_text())
 if meta.get('ee_delta')!='anchor minus current; rotation log(R_anchor_root @ R_current_root.T)':raise ValueError('Old or unknown EE direction: regenerate capture geometry before producing v2 teachers')
 scene=[];valids=[];coverages=[];timings=[]
 for cam in CAMS:
  paths=[capture/f'{cam}_current.png',capture/f'{cam}_anchor.png']
  images=[cv2.cvtColor(cv2.imread(str(p),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB) for p in paths]
  start=time.perf_counter()
  f,confidence,back,_,_=predict(paths[0],paths[1],images[0],images[1]);timings.append(time.perf_counter()-start)
  with np.load(capture/f'{cam}_masks.npz') as m:
   latent,mask,coverage,pixels=pool_match(f,m['current_background'],m['anchor_background'],back,confidence,confidence_threshold=confidence_threshold,fb_threshold_px=fb_threshold_px,min_valid_fraction=min_valid_fraction)
  scene.append(latent);valids.append(mask);coverages.append(coverage)
  if save_debug:
   output.mkdir(parents=True,exist_ok=True);np.savez_compressed(output/f'{cam}_debug.npz',flow_px=f,valid_pixels=pixels,coverage=coverage)
 with np.load(capture/'geometry.npz') as z:ee=z['ee_latent'];ee_valid=z['ee_valid']
 checkpoint=TEACHER_ROOT/'weights/romav2.0.1.pt'
 meta.update(roma_seed=0,roma_checkpoint_sha256=checkpoint_sha256(checkpoint),cache_version='roma_ee_window_v2_current_to_anchor',scene_direction='current_t_to_paired_anchor_frame_0',scene_units='du/width,dv/height',pooling='masked_average_4x4_left_then_right',confidence_threshold=confidence_threshold,fb_threshold_px=fb_threshold_px,min_valid_fraction=min_valid_fraction,roma_checkpoint=str(TEACHER_ROOT/'weights/romav2.0.1.pt'),model_info=model_info or {},inference_seconds=timings)
 output.mkdir(parents=True,exist_ok=True);target=output/'window.npz';tmp=output/'window.tmp.npz'
 np.savez_compressed(tmp,frame_index=np.array([meta['frame']],np.int64),scene_latent=np.concatenate(scene)[None],scene_valid=np.concatenate(valids)[None],ee_latent=ee[None],ee_valid=ee_valid[None],metadata=np.asarray(json.dumps(meta)))
 tmp.replace(target);return {'path':str(target),'scene_valid_dims':int(np.sum(valids)),'ee_valid_dims':int(ee_valid.sum()),'timings':timings}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--capture',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');ap.add_argument('--confidence-threshold',type=float,default=None);ap.add_argument('--fb-threshold-px',type=float,default=None);ap.add_argument('--min-valid-fraction',type=float,default=.1);ap.add_argument('--save-debug',action='store_true');a=ap.parse_args()
 model,predict,info=load_roma(a.device)
 print(json.dumps(process(a.capture,a.output,predict,confidence_threshold=a.confidence_threshold,fb_threshold_px=a.fb_threshold_px,min_valid_fraction=a.min_valid_fraction,model_info=info,save_debug=a.save_debug),indent=2))
if __name__=='__main__':main()
