"""Run SHARP and render its expected depth for one real calibrated image."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sharp.cli.predict import predict_image
from sharp.models import PredictorParams, create_predictor
from sharp.utils.gaussians import save_ply
from sharp.utils.gsplat import GSplatRenderer

p=argparse.ArgumentParser()
p.add_argument('--image',required=True); p.add_argument('--checkpoint',required=True)
p.add_argument('--output',required=True); p.add_argument('--fovy-deg',type=float,required=True)
p.add_argument('--device',default='cuda')
a=p.parse_args(); device=torch.device(a.device)
if device.type!='cuda' or not torch.cuda.is_available(): raise RuntimeError('CUDA is required for SHARP depth rendering')
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
im=np.asarray(Image.open(a.image).convert('RGB')); h,w=im.shape[:2]
f=h/(2*np.tan(np.deg2rad(a.fovy_deg)/2))
model=create_predictor(PredictorParams());model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True));model.eval().to(device)
g=predict_image(model,im,float(f),device)
save_ply(g,float(f),(h,w),out/'gaussians.ply')
k=torch.tensor([[[f,0,(w-1)/2],[0,f,(h-1)/2],[0,0,1]]],dtype=torch.float32,device=device)
with torch.no_grad(): result=GSplatRenderer()(g,torch.eye(4,device=device)[None],k,w,h)
depth=result.depth.detach().cpu().numpy().squeeze();alpha=result.alpha.detach().cpu().numpy().squeeze()
valid=np.isfinite(depth)&(depth>0)&(alpha>0.5)
np.savez_compressed(out/'depth_alpha.npz',depth=depth,alpha=alpha,valid=valid)
stats={'height':h,'width':w,'focal_pixels':float(f),'valid_fraction':float(valid.mean()),'depth_min':float(depth[valid].min()) if valid.any() else None,'depth_median':float(np.median(depth[valid])) if valid.any() else None,'depth_max':float(depth[valid].max()) if valid.any() else None,'coordinates':'OpenCV camera x-right y-down z-forward; depth in model metric scale'}
(out/'render_stats.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2))
if not valid.any(): raise RuntimeError('no valid rendered depth')
