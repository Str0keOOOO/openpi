"""Deterministic small offsets about the recorded official Anchor."""
import math

RADII_M=(0.,.005,.01,.02,.03,.05,.075,.10,.15)
YAW_DEG=(0.,-1.,1.,-2.,2.,-3.,3.,-5.,5.)
YAW_LEVER_M=.5


def offsets():
    candidates=[]
    for radius in RADII_M:
        for direction in range(1 if radius==0 else 16):
            phi=2*math.pi*direction/16
            for yaw in YAW_DEG:
                dx,dy=radius*math.cos(phi),radius*math.sin(phi)
                # Comparable SE(2) distance: translation plus yaw displacement
                # at a declared 0.5 m lever, not a claim of global optimality.
                score=math.hypot(radius,YAW_LEVER_M*math.radians(yaw))
                candidates.append({'offset':[dx,dy,yaw],'translation_m':radius,'distance_score_m':score})
    candidates.sort(key=lambda x:(x['distance_score_m'],abs(x['offset'][2]),x['offset'][0],x['offset'][1]))
    for i,c in enumerate(candidates):c['candidate_id']=f'candidate_{i:04d}'
    return candidates
