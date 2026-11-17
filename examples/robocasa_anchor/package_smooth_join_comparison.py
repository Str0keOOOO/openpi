"""Copy selected successful pilot videos to one user-facing comparison folder."""
import csv,json,shutil,hashlib
from pathlib import Path
ROOT=Path('/data5/xuchenfei/handoff_comparison')
SELECTED=[ROOT/'drawer_duration_1.65/OpenDrawer_ep000467',ROOT/'revised/OpenStandMixerHead_ep000441',ROOT/'revised/PickPlaceCounterToCabinet_ep000331',ROOT/'revised/PickPlaceSinkToCounter_ep000265']

def main():
    rows=[];selection=[]
    for source in SELECTED:
        s=json.loads((source/'summary.json').read_text());v=json.loads((source/'actual_states_validation.json').read_text())
        if s['status']!='task_success' or not v['passed']:raise ValueError(f'Unverified source {source}')
        with (source.parent/'measured_comparison.csv').open() as f:
            row=next(r for r in csv.DictReader(f) if r['task']==s['task'])
        destination=ROOT/'videos'/s['task'];destination.mkdir(parents=True,exist_ok=True)
        for name in ['old.mp4','new.mp4','human_reference.mp4','comparison.mp4','summary.json','actual_states_validation.json','video_validation.json']:
            shutil.copy2(source/name,destination/name)
        row['comparison_video']=str(destination/'comparison.mp4');rows.append(row)
        selection.append(dict(task=s['task'],source=str(source),destination=str(destination),requested_method='same-Human precontact join + quintic target + original OSC feedback',handoff_duration_scale=1.65 if s['task']=='OpenDrawer' else 1.0,notes='Small handoff reversal remains' if s['task']=='OpenDrawer' else 'Pilot successful'))
    with (ROOT/'comparison_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (ROOT/'final_selection.json').write_text(json.dumps(selection,indent=2))
    lines=['最终对比视频：本目录 videos/ 下每任务一个文件夹，共四组。','每组 comparison.mp4 左边旧版、右边新版；old.mp4、new.mp4 单独保存；human_reference.mp4 是配对原始演示。','所有视频按真实20Hz播放，没有视频加速。短视频结束后并排画面保持末帧，并注明ended。','方法：同一Human、同一场景、相同初始机器人状态，在第一次接触/闭爪/身体控制前选择衔接帧；跳过冗余自由空间往返，保留后续操作。五次多项式同时连接位置和姿态，并匹配接入后的运动方向。','四条实际执行均达到任务成功条件；已检查实际衔接段和操作段的关节限位（沿用轻微瞬时软限位容差）与非相邻自穿透记录；视频逐帧解码通过。','现有数据集没有修改，没有批量生成，没有训练。','这些结果说明小样本可减少路径与耗时，并不代表所有样本都已经没有往返。尤其抽屉接点仍有轻微回摆；未选用会破坏3mm接入门槛的持续前馈尝试。','']
    names={'OpenDrawer':'打开抽屉','OpenStandMixerHead':'打开搅拌机头','PickPlaceCounterToCabinet':'台面放柜子','PickPlaceSinkToCounter':'水槽放台面'}
    for r in rows:
        lines.append(f"{names[r['task']]}：旧{float(r['old_seconds']):.2f}秒 → 新{float(r['new_seconds']):.2f}秒；实际末端总路径减少{float(r['path_reduction_percent']):.1f}%；Human接入帧{r['source_join_frame']}。")
    lines+=['','目录保留了早期试验与失败记录，请优先看videos/。','代码快照在code/；当前正式实现文件在robocasa365/openpi/examples/robocasa_anchor/。','复现四条原始修正试验：bash /data5/xuchenfei/handoff_comparison/launch.sh','复现抽屉时长候选：bash /data5/xuchenfei/handoff_comparison/refine_drawer.sh']
    (ROOT/'先看这里.txt').write_text('\n'.join(lines)+'\n')
    code=Path('/data4/xuchenfei/robocasa365/third_party/openpi/examples/robocasa_anchor');snapshot=ROOT/'code_final';snapshot.mkdir(exist_ok=True)
    manifest=[]
    for name in ['smooth_join_plan.py','transition_smooth.py','preflight_smooth_join.py','run_smooth_join_comparison.py','summarize_smooth_join_comparison.py','verify_smooth_join_states.py','package_smooth_join_comparison.py','operation.py','soft_joint_limits.py','anchor_pose.py']:
        p=code/name;shutil.copy2(p,snapshot/name);manifest.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (snapshot/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(selected=len(rows),folder=str(ROOT/'videos'),dataset_modified=False,results=rows),indent=2))
if __name__=='__main__':main()
