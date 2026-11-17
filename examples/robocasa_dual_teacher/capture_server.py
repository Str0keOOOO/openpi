"""Keep a single MuJoCo scene alive for independent windows in one episode."""
import contextlib,gzip,hashlib,io,json,pickle,struct,sys,time,traceback
from collections import OrderedDict
from pathlib import Path
with contextlib.redirect_stdout(sys.stderr):
    from examples.robocasa_dual_teacher import capture_window as c

class Lease:
    def __init__(self,wrapped):self.wrapped=wrapped
    def __getattr__(self,key):return getattr(self.wrapped,key)
    def close(self):pass

def main():
    original_open,original_move,original_sha,original_seg,original_rgb=c._open_env,c._episode_moved_geoms,c.sha,c._geom_id_image,c.rgb
    holder={};moves={};hashes={};segmentations=OrderedDict();video_cache=OrderedDict();rgb_zero={};source_sets={};xml_keys={}
    def fast_rgb(ds,episode,camera,frame):
        import cv2
        p=ds/"videos"/f"chunk-{episode//1000:03d}"/f"observation.images.{camera}"/f"episode_{episode:06d}.mp4"
        key=str(p)
        if frame==0 and key in rgb_zero:return rgb_zero[key].copy(),p
        if key not in video_cache:
            while len(video_cache)>=6:
                _,(oldcap,_)=video_cache.popitem(last=False);oldcap.release()
            cap=cv2.VideoCapture(key)
            if not cap.isOpened():raise RuntimeError("Unreadable video "+key)
            video_cache[key]=(cap,0)
        video_cache.move_to_end(key)
        cap,nextframe=video_cache[key]
        if nextframe!=frame:cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
        ok,bgr=cap.read()
        if not ok:raise RuntimeError("Video decode failed "+key+" frame "+str(frame))
        image=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);video_cache[key]=(cap,frame+1)
        if frame==0:rgb_zero[key]=image.copy()
        return image,p
    def open_once(task,episode,root,meta,dataset_meta,states):
        xmlpath=root/'extras'/f'episode_{episode:06d}'/'model.xml.gz'
        signature=cached_sha(xmlpath)
        if signature not in xml_keys:xml_keys[signature]=hashlib.sha256(gzip.decompress(xmlpath.read_bytes())).hexdigest()
        key=(task,str(root.resolve()),xml_keys[signature])
        if 'env' not in holder:
            holder['key']=key;holder['env']=original_open(task,episode,root,meta,dataset_meta,states)
            sim=holder['env'].sim;original_render=sim.render;render_cache=OrderedDict()
            def cached_render(*args,**kwargs):
                key=(repr(args),repr(sorted(kwargs.items())),hashlib.sha256(sim.data.qpos.tobytes()).digest())
                if key not in render_cache:
                    render_cache[key]=original_render(*args,**kwargs).copy()
                    if len(render_cache)>12:render_cache.popitem(last=False)
                else:render_cache.move_to_end(key)
                return render_cache[key].copy()
            sim.render=cached_render
            files=[root/'extras'/f'episode_{episode:06d}'/'states.npz',root/'extras'/f'episode_{episode:06d}'/'model.xml.gz']
            holder['source_signatures']={str(p):original_sha(p) for p in files}
        if holder['key']!=key:raise ValueError('Grouped capture requires identical task and exact scene XML')
        for p,expected in holder['source_signatures'].items():
            if cached_sha(p)!=expected:raise ValueError('Source episode changed during cache generation: '+p)
        return Lease(holder['env'])
    def cached_sha(p):
        p=Path(p);st=p.stat();key=(str(p),st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
        if key not in hashes:hashes[key]=original_sha(p)
        return hashes[key]
    def moved(sim,states,robot_bodies,translation_threshold_m,rotation_threshold_deg):
        h=hashlib.sha256();h.update(str(states.shape).encode());h.update(states[0].tobytes());h.update(states[-1].tobytes())
        key=(h.digest(),translation_threshold_m,rotation_threshold_deg)
        if key not in moves:moves[key]=original_move(sim,states,robot_bodies,translation_threshold_m,rotation_threshold_deg)
        return moves[key]
    def seg(sim,camera):
        h=hashlib.sha256();h.update(sim.data.qpos.tobytes());key=(camera,h.digest())
        if key not in segmentations:
            segmentations[key]=original_seg(sim,camera).copy()
            if len(segmentations)>8:segmentations.popitem(last=False)
        else:segmentations.move_to_end(key)
        return segmentations[key]
    c._open_env=open_once;c._episode_moved_geoms=moved;c.sha=cached_sha;c._geom_id_image=seg;c.rgb=fast_rgb
    try:
        for line in sys.stdin:
            try:
                request=json.loads(line);start=time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):payload=c.main(request['args'],return_arrays=request.get('binary',False))
                if request.get('binary',False):meta=payload['metadata']
                else:
                    output_dir=Path(request['args'][request['args'].index('--output')+1]);meta=json.loads((output_dir/'metadata.json').read_text())
                signatures=meta['file_sha256'];source_key=(meta['task'],meta['episode'])
                if source_key not in source_sets:source_sets[source_key]=signatures
                elif source_sets[source_key]!=signatures:raise ValueError('Current or Anchor source changed during episode generation')
                result={'ok':True,'payload':payload,'capture_seconds':time.perf_counter()-start}
            except Exception as exc:
                result={'ok':False,'error':str(exc),'traceback':traceback.format_exc()}
            if request.get('binary',False):
                raw=pickle.dumps(result,protocol=5);sys.stdout.buffer.write(struct.pack('<Q',len(raw)));sys.stdout.buffer.write(raw);sys.stdout.buffer.flush()
            else:sys.stdout.write(json.dumps(result)+'\n');sys.stdout.flush()
    finally:
        for cap,_ in video_cache.values():cap.release()
        if 'env' in holder:holder['env'].close()
if __name__=='__main__':main()
