"""Native OpenPI RGB augmentation with exact affine transformation of dense flow.

Both endpoints use the same per-camera affine augmentation. RGB is identical
(up to numerical tolerance) to the original augmax chain and RNG. Masked linear
sampling rejects incomplete interpolation footprints. No RoMa runs here.
"""
import augmax
import jax
import jax.numpy as jnp
from openpi.models import model as model_lib


def pool(flow, valid, minimum_fraction=.1):
    b,h,w,_=flow.shape
    if h%4 or w%4:raise ValueError('Model resolution must be divisible by 4')
    f=flow.reshape(b,4,h//4,4,w//4,2)
    m=valid.reshape(b,4,h//4,4,w//4,1)
    counts=jnp.sum(m,axis=(2,4));sums=jnp.sum(jnp.where(m,f,0),axis=(2,4))
    good=(counts>=minimum_fraction*(h//4)*(w//4))&(counts>0)
    value=sums/jnp.maximum(counts,1)
    return jnp.where(good,value,0).reshape(b,32),jnp.broadcast_to(good,value.shape).reshape(b,32)


def camera_augmentation(image, flow, valid, rng, *, geometric=True, train=True):
    """image Bx224x224x3, flow BxHxWx2 normalized dx/W,dy/H."""
    b,h,w,_=image.shape
    weight=valid.astype(jnp.float32)
    packed=jnp.concatenate([jnp.where(valid[...,None],flow,0),weight[...,None]],axis=-1)
    packed=jax.image.resize(packed,(b,h,w,3),method='linear')
    points=jnp.array([[0.,0.],[1.,0.],[0.,1.]],jnp.float32) # augmax points: row,column
    if train:
        transforms=[]
        if geometric:transforms += [augmax.RandomCrop(int(w*.95),int(h*.95)),augmax.Resize(w,h),augmax.Rotate((-5,5))]
        transforms += [augmax.ColorJitter(brightness=.3,contrast=.4,saturation=.5)]
        types=(augmax.InputType.IMAGE,augmax.InputType.DENSE,augmax.InputType.KEYPOINTS)
        chain=augmax.Chain(*transforms,input_types=types)
        keys=jax.random.split(rng,b)
        rgb,dense,basis=jax.vmap(lambda key,im,fields:chain(key,(im,fields,points)))(keys,image/2+.5,packed)
        rgb=rgb*2-1
    else:
        rgb=image;dense=packed;basis=jnp.broadcast_to(points,(b,3,2))
    # p_aug = A p_original + translation. Apply A to the displacement vector.
    linear=jnp.stack([basis[:,1]-basis[:,0],basis[:,2]-basis[:,0]],axis=-1)
    good=dense[...,2]>.9999
    f=dense[...,:2]/jnp.maximum(dense[...,2:3],1e-8)
    vectors=jnp.stack([f[...,1]*h,f[...,0]*w],axis=-1)
    vectors=jnp.einsum('bij,bhwj->bhwi',linear,vectors)
    output=jnp.stack([vectors[...,1]/w,vectors[...,0]/h],axis=-1)
    yy,xx=jnp.mgrid[:h,:w];ex=xx+output[...,0]*w;ey=yy+output[...,1]*h
    good=good&jnp.isfinite(output).all(-1)&(ex>=0)&(ex<w-1)&(ey>=0)&(ey<h-1)
    return rgb,output,good,linear


def preprocess(rng, observation, scene_flow, scene_valid, *, train):
    if scene_flow.ndim!=5 or scene_flow.shape[1]!=2 or scene_flow.shape[-1]!=2:raise ValueError('Expected Bx2xHxWx2 flow')
    if scene_valid.shape!=scene_flow.shape[:-1]:raise ValueError('Dense scene mask shape mismatch')
    # Standard resizing and image masks; no random transform has run yet.
    obs=model_lib.preprocess_observation(None,observation,train=False)
    images=dict(obs.images);latents=[];masks=[]
    for i,key in enumerate(['base_0_rgb','right_wrist_0_rgb']):
        rgb,f,m,_=camera_augmentation(images[key],scene_flow[:,i],scene_valid[:,i],rng,geometric='wrist' not in key,train=train)
        images[key]=rgb;z,v=pool(f,m);latents.append(z);masks.append(v)
    if train:
        key='left_wrist_0_rgb';im=images[key]/2+.5
        chain=augmax.Chain(augmax.ColorJitter(brightness=.3,contrast=.4,saturation=.5))
        images[key]=jax.vmap(chain)(jax.random.split(rng,im.shape[0]),im)*2-1
    return obs.replace(images=images),jnp.concatenate(latents,-1),jnp.concatenate(masks,-1)
