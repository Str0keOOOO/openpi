"""Two real-state pilot comparisons; no production dataset writes."""
import concurrent.futures,json,os,shutil,subprocess,sys,time
from pathlib import Path
import cv2,numpy as np,imageio.v2 as imageio
from scipy.spatial.transform import Rotation
from run_smooth_join_comparison import merge
CODE=Path(__file__).parent;OUT=Path('/data5/xuchenfei/handoff_continuous_two');DATA=Path('/data4/xuchenfei/robocasa365/data/household_augmented');CAM='robot0_agentview_left'

def compare(old,new,destination,ep,oldn,newn):
    caps=[cv2.VideoCapture(str(old)),cv2.VideoCapture(str(new))];last=[None,None];done=[False,False];step=0
    w=imageio.get_writer(str(destination),fps=20,codec='libx264',pixelformat='yuv420p',macro_block_size=2,ffmpeg_params=['-threads','1','-preset','veryfast'])
    try:
        while True:
            any_new=False
            for j,c in enumerate(caps):
                if done[j]:continue
                ok,img=c.read()
                if ok:last[j]=img;any_new=True
                else:done[j]=True
            if not any_new:break
            frames=[]
            for j,img in enumerate(last):
                if img is None:raise ValueError('Empty comparison input')
                canvas=np.zeros((300,256,3),np.uint8);canvas[44:]=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                title=f'OLD EP{ep}' if j==0 else f'NEW continuous EP{ep}'
                phase=('bridge' if step<oldn else 'operation') if j==0 else ('continuous approach' if step<newn else 'operation')
                cv2.putText(canvas,title,(5,16),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1);cv2.putText(canvas,f'{step/20:.2f}s {phase}'+(' [ended]' if done[j] else ''),(5,35),cv2.FONT_HERSHEY_SIMPLEX,.37,(230,230,230),1);frames.append(canvas)
            w.append_data(np.concatenate(frames,axis=1));step+=1
    finally:
        w.close()
        for c in caps:c.release()

def motion_metrics(poses,join):
    p=poses[:,:3];v=np.diff(p,axis=0)*20;r=Rotation.from_quat(poses[:,3:]).as_matrix();omega=Rotation.from_matrix(r[1:]@r[:-1].transpose(0,2,1)).magnitude()*20
    lo=max(1,join-10);hi=min(len(v),join+10);low=(np.linalg.norm(v[lo:hi],axis=1)<.01)&(omega[lo:hi]<.05);delta=np.diff(np.r_[False,low,False].astype(int));lengths=np.flatnonzero(delta==-1)-np.flatnonzero(delta==1)
    before=np.mean(v[max(0,join-5):max(1,join)],axis=0);after=np.mean(v[join:min(len(v),join+5)],axis=0)
    denominator=np.linalg.norm(before)*np.linalg.norm(after)
    return dict(seconds=len(poses)/20,path_m=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()),join_mean_speed_before_m_s=float(np.linalg.norm(before)),join_mean_speed_after_m_s=float(np.linalg.norm(after)),join_direction_cosine=float(before@after/denominator) if denominator>1e-6 else None,join_velocity_change_m_s=float(np.linalg.norm(after-before)),join_low_motion_seconds=float(low.sum()/20),join_longest_low_motion_seconds=float(max(lengths,default=0)/20),definition='Join +/-0.5s; low motion requires both EE linear speed<1cm/s and angular speed<0.05rad/s; natural task holds are not classified as failures')

def run(ep,label):
    start=time.time();folder=OUT/f'OpenDrawer_ep{ep:06d}';folder.mkdir(parents=True,exist_ok=True);extra=DATA/f'OpenDrawer/lerobot/extras/episode_{ep:06d}';prov=json.loads((extra/'provenance.json').read_text());old=Path(f'/data5/xuchenfei/OpenDrawer_ep{ep:06d}_{label}.mp4');shutil.copy2(old,folder/'old.mp4')
    gpu=1;uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip();env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',__EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/10_nvidia.json')
    attempts=[];accepted=None
    for rank_index,(rank,scale) in enumerate([(0,1.35),(0,1.65),(1,1.35),(1,1.0)]):
        trial=folder/f'candidate_{rank_index}';trial.mkdir(exist_ok=True);cmd=[sys.executable,'-u',str(CODE/'preflight_continuous.py'),'--task','OpenDrawer','--episode',str(prov['source_human_episode_index']),'--gpu',str(gpu),'--output',str(trial),'--execute-transition','--execute-operation','--reference-anchor-extras',str(extra),'--restore-initial-extras',str(extra),'--delta11',*(['0']*11),'--handoff-rank',str(rank),'--handoff-duration-scale',str(scale)]
        (trial/'command.json').write_text(json.dumps(cmd,indent=2))
        with (trial/'run.log').open('w') as log:
            try:q=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
            except subprocess.TimeoutExpired:attempts.append(dict(rank=rank_index,status='timeout'));continue
        r=json.loads((trial/'result.json').read_text())
        if r.get('status')=='operation_success' and not r['transition'].get('collision_review_required') and not r['operation'].get('collision_review_required'):
            with np.load(trial/'transition_actual.npz') as z:
                with np.load(extra/'states.npz') as original:assert np.allclose(z['sim_state_before_action'][0],original['states'][0],atol=1e-9,rtol=0),'Initial state changed'
            accepted=(rank_index,trial,r);attempts.append(dict(rank=rank_index,status='accepted'));break
        reason=r.get('error') or r.get('failure_reason') or r.get('transition',{}).get('failure_reason') or ('contact review' if r.get('status')=='operation_success' else r.get('status'));attempts.append(dict(rank=rank_index,status=r.get('status'),reason=reason))
    if accepted is None:
        result=dict(task='OpenDrawer',episode=ep,human=prov['source_human_episode_index'],status='no_accepted_candidate',attempts=attempts,seconds=time.time()-start);(folder/'summary.json').write_text(json.dumps(result,indent=2));return result
    index,trial,r=accepted;tr=r['transition'];merge([trial/f'transition_{CAM}.mp4',trial/f'{CAM}.mp4'],folder/'new.mp4');compare(old,folder/'new.mp4',folder/'comparison.mp4',ep,int(prov['transition_frames']),int(tr['steps']))
    with np.load(extra/'ee_world.npz') as z:oldposes=z['ee_xyz_quat_xyzw']
    with np.load(trial/'transition_actual.npz') as z:bridge=z['ee_world_xyz_quat_xyzw'];tracking=z['tracking_error_after_action_m_rad'];rates=z['time_scale']
    with np.load(trial/'operation_actual.npz') as z:newposes=np.concatenate([bridge,z['ee_world_xyz_quat_xyzw']])
    result=dict(task='OpenDrawer',episode=ep,human=prov['source_human_episode_index'],status='task_success',attempts=attempts,seconds=time.time()-start,old=motion_metrics(oldposes,int(prov['transition_frames'])),new=motion_metrics(newposes,int(tr['virtual_join_step'])),virtual_join_step=tr['virtual_join_step'],virtual_source_join_frame=tr['virtual_join_frame'],retained_operation_start_frame=tr['source_resume_frame'],continuous_controller=tr['speed_profile'],free_space_tracking_position_error_p95_m=float(np.quantile(tracking[:,0],.95)),free_space_tracking_rotation_error_p95_rad=float(np.quantile(tracking[:,1],.95)),moving_boundary_position_error_m=tr['final_position_error_m'],moving_boundary_rotation_error_rad=tr['final_rotation_error_rad'],camera_video_rate_hz=20,comparison=str(folder/'comparison.mp4'),dataset_modified=False)
    (folder/'summary.json').write_text(json.dumps(result,indent=2))
    from verify_smooth_join_states import verify
    validation=verify(folder)
    if not validation['passed']:result['status']='actual_state_validation_failed';(folder/'summary.json').write_text(json.dumps(result,indent=2));return result
    destination=Path(f'/data5/xuchenfei/OpenDrawer_ep{ep:06d}_continuous_comparison.mp4');shutil.copy2(folder/'comparison.mp4',destination);result['delivered_comparison']=str(destination);(folder/'summary.json').write_text(json.dumps(result,indent=2));return result

def main():
    OUT.mkdir(exist_ok=True);results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for r in pool.map(lambda item:run(*item),[(467,'selected'),(470,'unselected')]):
            results.append(r);(OUT/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(r),flush=True)
    (OUT/'finished.json').write_text(json.dumps(dict(processed=len(results),task_success=sum(r['status']=='task_success' for r in results),dataset_modified=False),indent=2))
if __name__=='__main__':main()
