from model.rotation2xyz import Rotation2xyz
from trimesh import Trimesh
import torch
import utils.rotation_conversions as geometry

from visualize.simplify_loc2rot import joints2smpl
from visualize.converter import converter

class npy2obj(converter):
    def __init__(self, motion_dict, sample_idx, interpolate=1.0, device=0, cuda=True):
        motion = motion_dict['motion']
        bs, njoints, nfeats, nframes = motion.shape
        assert nfeats == 3
        
        rot2xyz = Rotation2xyz(device='cpu')
        self.faces = rot2xyz.smpl_model.faces
        
        self.num_frames = motion[sample_idx].shape[-1]
        j2s = joints2smpl(num_frames=self.num_frames, device_id=device, cuda=cuda)
        
        print(f'Running SMPLify For sample [{sample_idx}], it may take a few minutes.')
        motion_tensor, opt_dict = j2s.joint2smpl(motion[sample_idx].transpose(2, 0, 1))  # [nframes, njoints, 3]
        self.opt_dict = opt_dict
        
        motion = self.preprocess_motion(motion_tensor, opt_dict['cam'], interpolate).cpu().numpy()
        self.num_frames = motion.shape[-1]
        self.vertices = rot2xyz(torch.tensor(motion), mask=None,
                                     pose_rep='rot6d', translation=True, glob=True,
                                     jointstype='vertices',
                                     # jointstype='smpl',  # for joint locations
                                     vertstrans=True)
                                     
    def get_vertices(self, sample_i, frame_i):
        return self.vertices[sample_i, :, :, frame_i].squeeze().tolist()

    def get_trimesh(self, sample_i, frame_i):
        return Trimesh(vertices=self.get_vertices(sample_i, frame_i), faces=self.faces)
    
    def get_traj(self):
        root_positions = self.vertices.numpy().mean(axis=(0, 1))  # [3, frame_n]
        root_positions = root_positions.transpose(1, 0)  # [frame_n, 3]
        return root_positions

    def save_obj(self, save_path, frame_i):
        mesh = self.get_trimesh(0, frame_i)
        with open(save_path, 'w') as fw:
            mesh.export(fw, 'obj')
        return save_path     
    
    def preprocess_motion(self, motion_tensor, cam, interpolate):
        # Reshape motion tensor with permute to maintain correct order
        thetas = motion_tensor[:, :24, :6, :].reshape(1, 24, 6, -1)
        root_loc = torch.cat([cam, torch.zeros_like(cam)], dim=2) # n*1*6
        root_loc = root_loc.permute(1, 2, 0).reshape(1, 1, 6, -1) # 1*1*6*n
        
        n_frames = thetas.shape[-1]
        total_frames = int((n_frames - 1) * interpolate + 1)
        
        interp_thetas = torch.zeros((1, 24, 6, total_frames), device=thetas.device)
        interp_root_loc = torch.zeros((1, 1, 6, total_frames), device=root_loc.device)
        
        thetas_reshaped = thetas.permute(0, 3, 1, 2).reshape(-1, 6)
        matrices = geometry.rotation_6d_to_matrix(thetas_reshaped)
        matrices = matrices.reshape(1, n_frames, 24, 3, 3).permute(0, 2, 3, 4, 1)
        
        for k in range(total_frames):
            frame_pos = k / interpolate
            frame_idx = int(frame_pos)
            alpha = frame_pos - frame_idx
            
            if frame_idx >= n_frames - 1:
                # Handle last frame
                interp_thetas[..., k] = thetas[..., -1]
                interp_root_loc[..., k] = root_loc[..., -1]
            else:
                # Interpolate rotations
                R1 = matrices[..., frame_idx]
                R2 = matrices[..., frame_idx + 1]
                R = geometry.matrix_slerp(R1, R2, alpha)
                interp_thetas[..., k] = geometry.matrix_to_rotation_6d(R)
                
                # Interpolate root positions
                p1 = root_loc[..., frame_idx]
                p2 = root_loc[..., frame_idx + 1]
                interp_root_loc[..., k] = p1 + alpha * (p2 - p1)
                
        thetas = torch.cat([interp_thetas, interp_root_loc], dim=1)
        
        return thetas
            
        
    
    