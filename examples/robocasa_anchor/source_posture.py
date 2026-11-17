"""Bounded OSC nullspace posture targets; never write actual joint positions."""
import numpy as np
class SourcePosture:
    def __init__(self,ctrl,model,joint_names,max_step_rad=.04,limit_margin_rad=.035):
        self.ctrl=ctrl;self.ids=np.array([model.jnt_qposadr[model.joint(n).id] for n in joint_names],int)
        ranges=np.array([model.jnt_range[model.joint(n).id] for n in joint_names])
        self.low=ranges[:,0]+limit_margin_rad;self.high=ranges[:,1]-limit_margin_rad
        self.target=np.asarray(ctrl.initial_joint).copy();self.max_step=max_step_rad
        if self.target.shape!=self.ids.shape:raise ValueError('OSC nullspace joint mapping mismatch')
    def update(self,state,blend=1.):
        requested=np.asarray(state)[1+self.ids];requested=np.clip(requested,self.low,self.high)
        self.target+=np.clip(blend*(requested-self.target),-self.max_step,self.max_step)
        # update_initial_joints additionally resets the primary OSC goal; only
        # the documented nullspace target attribute is changed here.
        self.ctrl.initial_joint=self.target.copy()
