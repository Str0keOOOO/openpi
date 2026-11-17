"""Spatial teacher transforms; observations are read by the official Groot loader."""
import numpy as np


class SpatialInputs:
    """RoboCasa's runtime mapping plus teacher fields for the spatial train step."""
    def __init__(self, action_dim: int): self.action_dim=int(action_dim)
    def __call__(self,data):
        from openpi.policies.robocasa_policy import RobocasaInputs
        from openpi.models import model as model_lib
        out=RobocasaInputs(action_dim=self.action_dim,model_type=model_lib.ModelType.PI05)(data)
        if 'teacher_latent' in data or 'teacher_valid' in data:
            if 'teacher_latent' not in data or 'teacher_valid' not in data:
                raise ValueError('teacher_latent and teacher_valid must be supplied together')
            out['teacher_latent']=np.asarray(data['teacher_latent'],np.float32)
            out['teacher_valid']=np.asarray(data['teacher_valid'],bool)
        return out
