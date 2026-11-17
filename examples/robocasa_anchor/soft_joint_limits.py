"""Small, transient soft-limit acceptance; report physical excursions explicitly."""
import numpy as np

def check_soft_joint_limits(model,joint_names,states,*,angular_cap_deg=.6,max_consecutive_frames=6,strict_tolerance=.001):
    states=np.asarray(states)
    if states.ndim!=2 or states.shape[1]!=1+model.nq+model.nv or not np.isfinite(states).all():
        raise ValueError('Invalid saved simulator trajectory')
    records=[];rejected=[]
    for joint in joint_names:
        jid=model.joint(joint).id
        if not model.jnt_limited[jid]:continue
        addr=int(model.jnt_qposadr[jid]);lo,hi=model.jnt_range[jid];v=states[:,1+addr]
        # mjJNT_HINGE=3, mjJNT_SLIDE=2. No extra tolerance for linear joints.
        angular=int(model.jnt_type[jid])==3
        cap=max(strict_tolerance,np.deg2rad(angular_cap_deg)) if angular else strict_tolerance
        excess=np.maximum(np.maximum(lo-v,v-hi),0)
        over=excess>strict_tolerance
        change=np.diff(np.r_[False,over,False].astype(int));runs=np.flatnonzero(change==-1)-np.flatnonzero(change==1)
        longest=int(runs.max()) if len(runs) else 0
        okay=bool(excess.max()<=cap and longest<=max_consecutive_frames)
        rec=dict(joint=joint,unit='deg' if angular else 'mm',max_excess=float(excess.max()*(180/np.pi if angular else 1000)),frames_above_strict=int(over.sum()),longest_consecutive_frames=longest,accepted=okay,allowed_max_excess=angular_cap_deg if angular else strict_tolerance*1000)
        records.append(rec)
        if not okay:rejected.append(rec)
    return dict(accepted=not rejected,joints=records,rejected=rejected,policy=dict(angular_cap_deg=angular_cap_deg,max_consecutive_frames=max_consecutive_frames,strict_tolerance_rad_or_m=strict_tolerance,scope='recorded operation states; same scope as strict version',basis='Real 418-Human audit: common arm maxima below 0.6 deg and consecutive excursions at most 6 frames; rarer larger excursions remain rejected.'))
