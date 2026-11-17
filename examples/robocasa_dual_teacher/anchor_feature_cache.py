"""Exact inference-only memoization of repeated Anchor descriptors/refiner features."""
from collections import OrderedDict
import hashlib
import torch

def enable(model,predict,limit_bytes=1536*1024**2):
 cache=OrderedDict();state={'token':None,'calls':{},'bytes':0,'hits':0}
 def container_copy(x):
  # RoMa matcher replaces the last element of descriptor lists in place.
  # Reuse tensor values, but never share mutable list/dict containers.
  if isinstance(x,list):return [container_copy(v) for v in x]
  if isinstance(x,tuple):return tuple(container_copy(v) for v in x)
  if isinstance(x,dict):return {k:container_copy(v) for k,v in x.items()}
  return x
 def size(x):
  if torch.is_tensor(x):return x.numel()*x.element_size()
  if isinstance(x,dict):return sum(size(v) for v in x.values())
  return sum(size(v) for v in x) if isinstance(x,(list,tuple)) else 0
 for name,module in [('descriptor',model.f),('refiner',model.refiner_features)]:
  original=module.forward
  def wrapped(x,*args,_name=name,_original=original,**kwargs):
   n=state['calls'].get(_name,0);state['calls'][_name]=n+1
   if n%2==0:return _original(x,*args,**kwargs)
   key=(_name,state['token'],tuple(x.shape),str(x.dtype))
   if key in cache:state['hits']+=1;cache.move_to_end(key);return container_copy(cache[key])
   out=_original(x,*args,**kwargs);count=size(out)
   if count<=limit_bytes:
    while cache and state['bytes']+count>limit_bytes:
     _,old=cache.popitem(last=False);state['bytes']-=size(old)
    cache[key]=container_copy(out);state['bytes']+=count
   return out
  module.forward=wrapped
 def run(pa,pb,a,b):
  state['token']=hashlib.sha256(b.tobytes()).digest();state['calls']={}
  return predict(pa,pb,a,b)
 run.cache_state=state
 return run
