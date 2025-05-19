import torch
import numpy as np
import visualize.utils.rotation_conversions as geometry
from visualize.rotation2xyz import Rotation2xyz
from visualize.jnt2rot import joints2smpl

class jnt2rot_wrapper:
    def __init__(self, motion_dict, sample_idx, device=0, cuda=True):
        motion = motion_dict['motion']
        bs, njoints, nfeats, nframes = motion.shape
        assert nfeats == 3
        
        self.original_num_frames = motion[sample_idx].shape[-1]
        j2s = joints2smpl(num_frames=self.original_num_frames, device_id=device, cuda=cuda)
        
        print(f'Running SMPLify For sample [{sample_idx}], it may take a few minutes.')
        motion_tensor, opt_dict = j2s.joint2smpl(motion[sample_idx].transpose(2, 0, 1))  # [nframes, njoints, 3]
        self.opt_dict = opt_dict
        
        self.motion_tensor = self.format_motion(motion_tensor, opt_dict['cam'])
        
    def format_motion(self, motion_tensor, cam):
        # Convert 6D rotations to 9D matrix form
        thetas = motion_tensor[:, :24, :6, :]  # [1, 24, 6, n]
        thetas = thetas.permute(0, 3, 1, 2)  # [1, n, 24, 6]
        thetas = geometry.rotation_6d_to_matrix(thetas)  # [1, n, 24, 3, 3]
        thetas = thetas.permute(0, 2, 3, 4, 1)  # [1, 24, 3, 3, n]
        thetas = thetas.reshape(1, 24, 9, -1)  # [1, 24, 9, n]
        
        # Handle root location (cam)
        root_loc = cam.permute(1, 2, 0).reshape(1, 1, 3, -1)  # n*1*3 to 1*1*3*n
        zeros = torch.zeros((1, 1, 6, root_loc.shape[-1]), device=root_loc.device)  # 1*1*6*n
        root_loc = torch.cat([root_loc, zeros], dim=2)  # 1*1*9*n
        
        thetas = torch.cat([thetas, root_loc], dim=1)  # [1, 25, 9, n]
        return thetas
    
    def get_motion_tensor(self):
        return self.motion_tensor
    
    def get_opt_dict(self):
        return self.opt_dict 