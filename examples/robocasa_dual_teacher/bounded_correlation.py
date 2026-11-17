"""Equivalent RoMa native local correlation evaluated in bounded row tiles."""
import torch
import torch.nn.functional as F

def native_torch_local_corr(feature0,feature1,warp,local_window,B,K,c,r,h,w,device,padding_mode='zeros',sample_mode='bilinear',dtype=torch.float32):
 out=torch.empty((B,K,h,w),device=device,dtype=dtype)
 for b in range(B):
  for start in range(0,h,16):
   end=min(start+16,h);rows=end-start
   coords=(warp[b,start:end,:,None,:]+local_window[:,None,None]).reshape(1,rows,w*K,2)
   window=F.grid_sample(feature1[b:b+1],coords,padding_mode=padding_mode,align_corners=False,mode=sample_mode).reshape(c,rows,w,K)
   out[b,:,start:end]=(feature0[b,:,start:end,:,None]/(c**.5)*window).sum(0).permute(2,0,1)
 return out

def install():
 import importlib
 module=importlib.import_module('romav2.local_correlation')
 module.native_torch_local_corr=native_torch_local_corr
