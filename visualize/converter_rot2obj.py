from trimesh import Trimesh
import torch
import math 
from visualize.converter import converter
from visualize.smooth import smooth_motion
from visualize.rotation2xyz import Rotation2xyz
import visualize.utils.rotation_conversions as geometry

class converter_rot2obj(converter):
    def __init__(self, motion_tensor, interpolate=1.0, device=0, cuda=True):
        # Initialize rotation to xyz converter
        rot2xyz = Rotation2xyz(device=motion_tensor.device)
        self.faces = rot2xyz.smpl_model.faces
        
        self.original_num_frames = motion_tensor.shape[-1]
        motion_tensor = self.postprocess_neck(motion_tensor)
        motion_tensor = smooth_motion(motion_tensor)
        self.interpolate = interpolate
        self.num_frames = int(self.original_num_frames * interpolate)
        
        self.vertices = rot2xyz(motion_tensor, mask=None,
                                pose_rep='rotmat', translation=True, glob=True,
                                jointstype='vertices',
                                # jointstype='smpl',  # for joint locations
                                vertstrans=True)
                                     
    def postprocess_neck(self, motion_tensor):
        rotations = motion_tensor[:,:-1] # shape [1, 24, 9, 104] (matrix)
        neck_joint_idx, head_joint_idx = 12, 15 # note neck is not 14 !!
        lwrist_joint_idx, rwrist_joint_idx = 20, 21 # palm is 22 23
        batch_size, joints, _, seq_len = rotations.shape
        
        # Identity rotation matrix
        identity_mat = torch.eye(3, dtype=rotations.dtype, device=rotations.device).view(1, 1, 9, 1)
        
        # Expand to [batch, 1, 9, seq_len] so it can be assigned per joint
        identity_mat_all = identity_mat.expand(batch_size, 1, 9, seq_len)

        for joint_idx in [12, 15, 20, 21, 22, 23]:
            rotations[:, joint_idx:joint_idx+1, :, :] = identity_mat_all
        
        motion_tensor[:,:-1] = rotations
        return motion_tensor
    
    def get_vertices(self, sample_i, frame_i):
        if self.interpolate == 1.0:
            return self.vertices[sample_i, :, :, frame_i].squeeze().tolist()
        else:
            # Get interpolated frame index
            frame_pos = frame_i / self.interpolate
            frame_idx = int(frame_pos)
            alpha = frame_pos - frame_idx
            
            if frame_idx >= self.original_num_frames - 1:
                # Handle last frame
                return self.vertices[sample_i, :, :, -1].squeeze().tolist()
            else:
                v1 = self.vertices[sample_i, :, :, frame_idx]
                v2 = self.vertices[sample_i, :, :, frame_idx + 1]
                interp = v1 + alpha * (v2 - v1)
                return interp.squeeze().tolist()

    def get_trimesh(self, sample_i, frame_i):
        return Trimesh(vertices=self.get_vertices(sample_i, frame_i), faces=self.faces)
    
    def get_traj(self):
        root_positions = self.vertices.cpu().numpy().mean(axis=(0, 1))  # [3, frame_n]
        root_positions = root_positions.transpose(1, 0)  # [frame_n, 3]
        return root_positions

    def save_obj(self, save_path, frame_i):
        mesh = self.get_trimesh(0, frame_i)
        with open(save_path, 'w') as fw:
            mesh.export(fw, 'obj')
        return save_path     
    
    def format_motion(self, motion_tensor, cam):
        # Reshape motion tensor with permute to maintain correct order
        thetas = motion_tensor[:, :24, :6, :].reshape(1, 24, 6, -1)
        root_loc = torch.cat([cam, torch.zeros_like(cam)], dim=2) # n*1*6
        root_loc = root_loc.permute(1, 2, 0).reshape(1, 1, 6, -1) # 1*1*6*n        
        thetas = torch.cat([thetas, root_loc], dim=1)
        
        return thetas
            
        
    
    