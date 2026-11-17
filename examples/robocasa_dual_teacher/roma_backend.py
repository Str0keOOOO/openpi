"""Local frozen RoMa v2 backend; independent of archived model comparisons."""
from pathlib import Path
import os,sys
import cv2,numpy as np,torch
ROOT=Path(os.environ.get('ROMA_TEACHER_ROOT',Path(__file__).resolve().parents[4]/'model/roma_v2'))

def resize_flow(flow,h,w):
    sh,sw=flow.shape[:2]
    if (sh,sw)!=(h,w):
        flow=cv2.resize(flow,(w,h),interpolation=cv2.INTER_LINEAR)
        flow[...,0]*=w/sw;flow[...,1]*=h/sh
    return np.asarray(flow,dtype=np.float32)

def setup_roma(*, setting='fast', bidirectional=None):
    if setting not in ('turbo','fast','base','precise'):
        raise ValueError('Unknown official RoMa preset')
    repo=ROOT/"vendor/romav2"
    sys.path.insert(0,str(repo/"src"))
    weights=ROOT/"weights/romav2.0.1.pt"
    original=torch.hub.load_state_dict_from_url
    original_hub_load=torch.hub.load
    def local_state(url,*args,**kwargs):
        return torch.load(weights,map_location=kwargs.get("map_location","cpu"),weights_only=True)
    def local_hub_load(repo_or_dir,*args,**kwargs):
        # RoMa v2 pins DINOv3 code at this exact commit. Its official RoMa checkpoint
        # supplies the descriptor weights, so only the pinned source tree is needed here.
        if isinstance(repo_or_dir,str) and repo_or_dir.startswith("facebookresearch/dinov3:"):
            kwargs.pop("skip_validation",None)
            return original_hub_load(str(ROOT/"vendor/dinov3"),*args,source="local",**kwargs)
        return original_hub_load(repo_or_dir,*args,**kwargs)
    torch.hub.load_state_dict_from_url=local_state
    torch.hub.load=local_hub_load
    from romav2 import RoMaV2
    torch.set_float32_matmul_precision("highest")
    try:
        model=RoMaV2()
    finally:
        torch.hub.load_state_dict_from_url=original
        torch.hub.load=original_hub_load
    model.apply_setting(setting)
    if bidirectional is not None and model.bidirectional != bidirectional:
        raise ValueError('Use the official preset without overriding bidirectional')
    model.eval().cpu()
    def predict(pa,pb,a,b):
        # Official array input avoids encoding and decoding temporary PNG files.
        with torch.inference_mode(): pred=model.match(a,b)
        def to_flow(warp,conf):
            if warp is None:
                return None,None,None
            warp=warp[0].detach().cpu().numpy()
            hp,wp=warp.shape[:2]
            x=(np.arange(wp,dtype=np.float32)+.5)*2/wp-1
            y=(np.arange(hp,dtype=np.float32)+.5)*2/hp-1
            xn,yn=np.meshgrid(x,y)
            dx=(warp[...,0]-xn)*(wp/2)
            dy=(warp[...,1]-yn)*(hp/2)
            f=resize_flow(np.stack((dx,dy),axis=-1),a.shape[0],a.shape[1])
            c=conf[0].detach().cpu().numpy()
            if c.ndim==3:c=c[...,0]
            c=cv2.resize(c,(a.shape[1],a.shape[0]),interpolation=cv2.INTER_LINEAR)
            return f,c,(hp,wp)
        fab,cab,sh=to_flow(pred["warp_AB"],pred["overlap_AB"])
        fba,cba,_=to_flow(pred["warp_BA"],pred["overlap_BA"])
        return fab,cab,fba,cba,sh
    info={"checkpoint":str(weights),"checkpoint_bytes":weights.stat().st_size,
          "setting":setting,"input_resolution":[256,256],"bidirectional":model.bidirectional,
          "internal_lr_resolution":[model.H_lr,model.W_lr],"internal_hr_resolution":[model.H_hr,model.W_hr],
          "inference_implementation":"official_model_match","extra_confidence_threshold":None,
          "forward_backward_filter":False if not model.bidirectional else None}
    return model,predict,info
