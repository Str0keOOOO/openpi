"""Official Fast with exact-scene reuse, two-frame prefetch and asynchronous saving."""
import argparse,concurrent.futures,fcntl,io,json,os,pickle,struct,subprocess,time,traceback,zipfile
from collections import deque
from pathlib import Path
import cv2,numpy as np,torch
from examples.robocasa_dual_teacher.roma_scene import load_roma,checkpoint_sha256,TEACHER_ROOT
from examples.robocasa_dual_teacher.geometry import pool_match
from examples.robocasa_dual_teacher.provenance import verify_sources
from examples.robocasa_dual_teacher.cache_schema import VERSION,ROMA_SETTING,BIDIRECTIONAL,validate_model_info,validate_metadata
from examples.robocasa_dual_teacher.state_ee import load_teacher,EE_VERSION
from examples.robocasa_augmentation.gpu_routing import egl_device_for_uuid

CAMS=('robot0_agentview_left','robot0_agentview_right')

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

class CaptureClient:
    def __init__(self,dataset,work,gpu_uuid,group_id):
        self.dataset=Path(dataset);self.work=Path(work);egl=egl_device_for_uuid(gpu_uuid)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(egl),MUJOCO_EGL_DEVICE_ID=str(egl),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',__EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/10_nvidia.json')
        logdir=self.work/'capture_logs';logdir.mkdir(parents=True,exist_ok=True);self.log=(logdir/f'{group_id}.log').open('a')
        self.process=subprocess.Popen(['/data4/xuchenfei/robocasa365/envs/robocasa/bin/python','-u','-m','examples.robocasa_dual_teacher.capture_server'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,bufsize=0,env=env)
    def read_exact(self,n):
        chunks=[]
        while n:
            block=self.process.stdout.read(n)
            if not block:raise RuntimeError(f'Mask process ended: {self.process.poll()}')
            chunks.append(block);n-=len(block)
        return b''.join(chunks)
    def capture(self,task,episode,frame):
        args=['--dataset',str(self.dataset),'--task',task,'--episode',str(episode),'--frame',str(frame),'--output',str(self.work/'unused_stream_output'),'--mask-policy','robot_only']
        self.process.stdin.write((json.dumps(dict(args=args,binary=True))+'\n').encode());self.process.stdin.flush()
        size=struct.unpack('<Q',self.read_exact(8))[0]
        if size>32*1024**2:raise ValueError('Unexpected capture message size')
        response=pickle.loads(self.read_exact(size)) # trusted local child process only
        if not response['ok']:raise RuntimeError(response.get('traceback',response.get('error')))
        response['payload']['capture_seconds']=response['capture_seconds'];return response['payload']
    def close(self):
        try:self.process.stdin.close();self.process.wait(timeout=15)
        except Exception:
            self.process.terminate()
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        self.log.close()

def save_frame(target,arrays):
    temporary=target.with_suffix('.tmp.npz')
    with temporary.open('wb') as stream:
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
            for name,value in arrays.items():
                buffer=io.BytesIO();np.lib.format.write_array(buffer,np.asarray(value),allow_pickle=False)
                archive.writestr(name+'.npy',buffer.getvalue())
    temporary.replace(target);return target.stat().st_size

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--uuid',required=True);p.add_argument('--slot',type=int,default=0);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    a.work.mkdir(parents=True,exist_ok=True)
    def draining():return (a.work/f'drain_gpu{a.gpu}').exists() or (a.work/f'drain_gpu{a.gpu}_slot{a.slot}').exists()
    cv2.setNumThreads(1);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.12,0)
    model,predict,info=load_roma('cuda',setting=ROMA_SETTING,bidirectional=BIDIRECTIONAL);validate_model_info(info)
    import romav2.local_correlation as lc
    if lc.native_torch_local_corr.__module__!='romav2.local_correlation' or lc.local_corr is not None:raise ValueError('Expected unmodified official native correlation')
    cksha=checkpoint_sha256(TEACHER_ROOT/'weights/romav2.0.1.pt')
    groups=json.loads((a.work/'groups.json').read_text());processed=0;began=time.time();workerid=f'gpu{a.gpu}_pid{os.getpid()}'
    lockdir=a.work/'group_locks';lockdir.mkdir(exist_ok=True)
    while True:
        pending=False;claimed=False
        for group in groups:
            if draining() and not a.pilot:break
            def folder(j):return a.cache/j['task']/f"episode_{j['episode']:06d}"
            todo=[j for j in group['jobs'] if not (folder(j)/'complete.json').exists() and not (folder(j)/'failed.json').exists()]
            if not todo:continue
            pending=True;lock=(lockdir/(group['id']+'.lock')).open('a')
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:lock.close();continue
            claimed=True;client=None
            try:
                client=CaptureClient(a.dataset,a.work,a.uuid,group['id'])
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as captures,concurrent.futures.ThreadPoolExecutor(max_workers=1) as saves:
                    for job in todo:
                        if draining() and not a.pilot:break
                        task,ep,n=job['task'],int(job['episode']),int(job['frames']);out=folder(job);out.mkdir(parents=True,exist_ok=True)
                        if (out/'complete.json').exists():continue
                        t=-1;saved=0;pending_save=None;queue=deque();episode_start=time.time()
                        try:
                            ee_values,ee_valid,em=load_teacher(out)
                            if len(ee_values)!=n or (em['task'],em['episode'],em['anchor_episode'])!=(task,ep,group['anchor_episode']) or em['robot_frame_proof']['xml_text_sha256']!=group['xml_text_sha256']:raise ValueError('Group/State EE provenance mismatch')
                            meta_path=out/'metadata.json'
                            if meta_path.exists():
                                meta=json.loads(meta_path.read_text());validate_metadata(meta);verify_sources(meta,task,ep,a.dataset/task/'lerobot')
                                if meta['roma_checkpoint_sha256']!=cksha:raise ValueError('Checkpoint changed')
                            frames=list(range(min(3,n))) if a.pilot else list(range(n));missing=[]
                            for t in frames:
                                target=out/f'frame_{t:06d}.npz'
                                if target.exists():
                                    with np.load(target,allow_pickle=False) as z:
                                        if int(z['frame_index'])!=t or str(z['cache_version'])!=VERSION:raise ValueError('Mixed saved-frame version')
                                    saved+=1
                                else:missing.append(t)
                            iterator=iter(missing)
                            def enqueue():
                                q=next(iterator,None)
                                if q is not None:queue.append((q,captures.submit(client.capture,task,ep,q)))
                            enqueue();enqueue()
                            def commit(item):
                                nonlocal saved,processed
                                future,record=item;size=future.result();saved+=1;processed+=1
                                record.update(status='running',gpu=a.gpu,pid=os.getpid(),task=task,episode=ep,episode_saved=saved,frames_processed=processed,elapsed_seconds=time.time()-began,last_cache_bytes=size,torch_peak_mib=torch.cuda.max_memory_allocated()/2**20)
                                write(a.work/f'{workerid}.json',record);write(out/'progress.json',dict(frames=n,saved=saved,failed=0,last_frame=record['frame'],worker=workerid));print(json.dumps(record),flush=True)
                            while queue and not (draining() and not a.pilot):
                                t,future=queue.popleft();start=time.perf_counter()
                                try:payload=future.result()
                                except Exception:
                                    for retry in range(2):
                                        try:payload=captures.submit(client.capture,task,ep,t).result();break
                                        except Exception:
                                            if retry==1:raise
                                enqueue()
                                if pending_save is not None:
                                    item=pending_save;pending_save=None;commit(item)
                                cm=payload['metadata']
                                if (cm['task'],cm['episode'],cm['frame'],cm['anchor_episode'])!=(task,ep,t,group['anchor_episode']):raise ValueError('Wrong prefetched frame')
                                fields=[];bits=[];robot_bits=[];anchor_bits=[];overlaps=[];coverage=[];quant=[];mt=time.perf_counter()
                                for cam in CAMS:
                                    im=payload['cameras'][cam];flow,confidence,back,_,_=predict(None,None,im['current'],im['anchor'])
                                    if back is not None or not np.isfinite(flow).all():raise ValueError('Unexpected official Fast output')
                                    _,_,_,valid=pool_match(flow,im['current_background'],im['anchor_background'],None,confidence,confidence_threshold=None,fb_threshold_px=None)
                                    normalized=flow/np.array([256,256],np.float32)
                                    # Only the agreed robot/boundary mask is applied. No confidence
                                    # or backward-flow rejection is used, so every usable forward
                                    # correspondence is preserved for the training target.
                                    normalized=np.where(valid[...,None],normalized,0);half=normalized.astype(np.float16)
                                    quant.append(float(np.max(np.abs(half.astype(np.float32)-normalized)))*256);fields.append(half);bits.append(np.packbits(valid.reshape(-1)));coverage.append(float(valid.mean()))
                                    robot_bits.append(np.packbits(im['current_background'].reshape(-1)));anchor_bits.append(np.packbits(im['anchor_background'].reshape(-1)))
                                match_seconds=time.perf_counter()-mt;ee,ev=ee_values[t],ee_valid[t]
                                if not np.allclose(payload['ee_latent'],ee,atol=1e-6,rtol=1e-6) or not np.array_equal(payload['ee_valid'],ev):raise ValueError('Recorded State EE mismatch')
                                if not meta_path.exists():
                                    cm.update(cache_version=VERSION,roma_checkpoint=str(TEACHER_ROOT/'weights/romav2.0.1.pt'),roma_checkpoint_sha256=cksha,roma_seed=0,roma_setting=ROMA_SETTING,roma_bidirectional=False,model_info=info,confidence_threshold=None,fb_threshold_px=None,min_valid_fraction=.1,source_frames=n,scene_direction='current_t_to_paired_anchor_frame_0',scene_units='du/width,dv/height',pooling='online_native_RGB_affine_then_masked_average_4x4',flow_storage='all finite in-bounds non-robot forward correspondences; zero outside the agreed mask',correlation_backends_used=['official_native_pytorch'],scene_group=group['id'])
                                    validate_metadata(cm);write(meta_path,cm)
                                arrays=dict(frame_index=np.asarray(t,np.int32),scene_flow=np.stack(fields),scene_valid_bits=np.stack(bits),current_background_bits=np.stack(robot_bits),anchor_background_bits=np.stack(anchor_bits),ee_latent=ee,ee_valid=ev,valid_pixel_ratio=np.asarray(coverage,np.float32),cache_version=np.asarray(VERSION),roma_setting=np.asarray(ROMA_SETTING),ee_source=np.asarray(EE_VERSION))
                                record=dict(frame=t,capture_seconds=payload['capture_seconds'],roma_seconds_total=match_seconds,current_window_seconds=time.perf_counter()-start,max_flow_fp16_error_px=max(quant),group_id=group['id'],pipeline='prefetch2_save1_official_fast')
                                pending_save=(saves.submit(save_frame,out/f'frame_{t:06d}.npz',arrays),record);(out/f'error_{t:06d}.json').unlink(missing_ok=True)
                            if pending_save is not None:
                                item=pending_save;pending_save=None;commit(item)
                            for _,future in queue:future.result()
                            queue.clear()
                            if saved==n and not a.pilot:write(out/'complete.json',dict(status='complete',frames=n,cache_version=VERSION,elapsed_seconds=time.time()-episode_start))
                            if a.pilot:write(out/'pilot.json',dict(status='passed',saved=saved,frames=n))
                        except Exception:
                            error=traceback.format_exc()
                            if pending_save is not None:
                                try:commit(pending_save)
                                except Exception:pass
                            for _,future in queue:
                                try:future.result()
                                except Exception:pass
                            write(out/'failed.json',dict(status='failed',frame=t,error=error,saved=saved));print(error,flush=True)
                if a.pilot:break
            finally:
                if client:client.close()
                fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        if a.pilot or not pending or draining():break
        if not claimed:time.sleep(5)
    write(a.work/f'{workerid}_finished.json',dict(status='finished',frames=processed,elapsed_seconds=time.time()-began))

if __name__=='__main__':main()
