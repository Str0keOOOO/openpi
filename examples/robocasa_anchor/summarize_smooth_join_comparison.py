"""Summarize measured pilot paths and check real-time comparison video integrity."""
import csv,json
from pathlib import Path
import cv2,numpy as np
ROOT=Path('/data5/xuchenfei/handoff_comparison/revised')
DATA=Path('/data4/xuchenfei/robocasa365/data/household_augmented')

def cosine(p,n):
    # Average velocity over 0.25 seconds on each side, to reduce timestep noise.
    before=p[n-1]-p[max(0,n-6)]; after=p[min(len(p)-1,n+4)]-p[n]
    denom=np.linalg.norm(before)*np.linalg.norm(after)
    return float(before@after/denom) if denom>1e-6 else None

def main():
    rows=[]
    for folder in sorted(ROOT.glob('*_ep*')):
        if not (folder/'summary.json').exists():continue
        r=json.loads((folder/'summary.json').read_text())
        if r['status']!='task_success':continue
        rank=next(x['rank'] for x in r['attempts'] if x['status']=='accepted'); trial=folder/f'candidate_{rank}'
        extra=DATA/r['task']/f"lerobot/extras/episode_{r['episode']:06d}"
        prov=json.loads((extra/'provenance.json').read_text());old=np.load(extra/'ee_world.npz')['ee_xyz_quat_xyzw'][:,:3]
        with np.load(trial/'transition_actual.npz') as z:tr=z['ee_world_xyz_quat_xyzw'][:,:3]
        with np.load(trial/'operation_actual.npz') as z:
            op=z['ee_world_xyz_quat_xyzw'][:,:3];err=z['tracking_error_after_action'] if 'tracking_error_after_action' in z.files else None
        new=np.concatenate([tr,op]);video_meta={}
        for name,expected in [('old',len(old)),('new',len(new)),('comparison',max(len(old),len(new)))]:
            cap=cv2.VideoCapture(str(folder/f'{name}.mp4'));fps=cap.get(cv2.CAP_PROP_FPS);frames=0
            while True:
                ok,frame=cap.read()
                if not ok:break
                if frame is None:raise ValueError('Missing decoded video frame')
                frames+=1
            cap.release()
            if frames!=expected or abs(fps-20)>1e-4:raise ValueError(f'Video mismatch {folder}/{name}: {frames}/{expected}, {fps}')
            video_meta[name]=dict(frames=frames,fps=fps,seconds=frames/fps)
        row=dict(task=r['task'],augmented_episode=r['episode'],source_human_episode=r['human'],source_join_frame=r['source_join_frame'],task_success=True,old_seconds=len(old)/20,new_seconds=len(new)/20,old_path_m=r['old_total_actual_path_m'],new_path_m=r['new_total_actual_path_m'],path_reduction_percent=100*(1-r['new_total_actual_path_m']/r['old_total_actual_path_m']),old_join_direction_cosine=cosine(old,int(prov['transition_frames'])),new_join_direction_cosine=cosine(new,len(tr)),transition_position_error_mm=1000*r['final_position_error_m'],transition_rotation_error_deg=float(np.degrees(r['final_rotation_error_rad'])),comparison_video=str(folder/'comparison.mp4'))
        rows.append(row)
        (folder/'video_validation.json').write_text(json.dumps(video_meta,indent=2))
    if rows:
        with (ROOT/'measured_comparison.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    text=['本目录为四条独立小样本真实仿真对比，不修改现有数据集。','comparison.mp4：左旧右新；同为真实20Hz，未加速；较短一侧结束后保持末帧。','old.mp4：原增强轨迹；new.mp4：新衔接实际执行；human_reference.mp4：对应原始Human。','同一场景、同一初始MuJoCo状态；在第一次接触、闭爪或身体控制之前选取接入帧，只替换自由空间接近段。','夹爪自由张开可在新接近段执行；抓取和后续操作保留。候选须完成任务并通过现有跟踪和接触检查。','本方法是同一演示内部的接入点优化，不是跨场景拼接，也不是MinInter原论文的直接复现。','方向余弦比较接点前后各0.25秒的平均位移：正值同向，负值反向，接近静止时不报告。','路径和时间统计截止各自首次任务成功；任务成功不代表所有未建模风险均已排除。','首次尝试位于上级文件夹，因未改善往返而未选作最终方案；本revised目录为修正后结果。','']
    for r in rows:
        text.append(f"{r['task']} EP{r['augmented_episode']}（Human EP{r['source_human_episode']}），接入Human第{r['source_join_frame']}帧：旧{r['old_seconds']:.2f}s，新{r['new_seconds']:.2f}s；实际EE路径旧{r['old_path_m']:.3f}m，新{r['new_path_m']:.3f}m，减少{r['path_reduction_percent']:.1f}%；接点方向余弦旧{r['old_join_direction_cosine']}，新{r['new_join_direction_cosine']}。")
    (ROOT/'说明与结果.txt').write_text('\n'.join(text)+'\n')
    print(json.dumps(rows,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
