from model.rotation2xyz import Rotation2xyz
import numpy as np
import trimesh
from trimesh import Trimesh
import os
import torch
from visualize.simplify_loc2rot import joints2smpl
import natsort
from pathlib import Path
import utils.rotation_conversions as geometry

class npy2obj:
    def __init__(self, motion_dict, sample_idx, rep_idx, interpolate=1.0, device=0, cuda=True):
        motion = motion_dict['motion']
        self.rot2xyz = Rotation2xyz(device='cpu')
        self.faces = self.rot2xyz.smpl_model.faces
        self.bs, self.njoints, self.nfeats, self.nframes = motion.shape
        self.opt_cache = {}
        self.sample_idx = sample_idx
        self.total_num_samples = motion_dict['num_samples']
        self.rep_idx = rep_idx
        self.absl_idx = self.sample_idx*self.total_num_samples + self.rep_idx
        self.num_frames = motion[self.absl_idx].shape[-1]
        self.j2s = joints2smpl(num_frames=self.num_frames, device_id=device, cuda=cuda)

        if self.nfeats == 3:
            print(f'Running SMPLify For sample [{sample_idx}], repetition [{rep_idx}], it may take a few minutes.')
            motion_tensor, opt_dict = self.j2s.joint2smpl(motion[self.absl_idx].transpose(2, 0, 1))  # [nframes, njoints, 3]
            # motion = motion_tensor.cpu().numpy()
            motion = self.interpolate_motions(motion_tensor, interpolate).cpu().numpy()
            self.num_frames = motion.shape[-1]
            
        elif self.nfeats == 6:
            motion = motion[[self.absl_idx]]
            
        
        self.bs, self.njoints, self.nfeats, self.nframes = motion.shape
        self.real_num_frames = motion_dict['lengths'][self.absl_idx] # 196
        
        self.vertices = self.rot2xyz(torch.tensor(motion), mask=None,
                                     pose_rep='rot6d', translation=True, glob=True,
                                     jointstype='vertices',
                                     # jointstype='smpl',  # for joint locations
                                     vertstrans=True)
        self.root_loc = motion[:, -1, :3, :].reshape(1, 1, 3, -1)

    def get_vertices(self, sample_i, frame_i, offset=None):
        if offset is not None:
            return self.vertices[sample_i, :, :, frame_i].squeeze().tolist() + offset
        else:
            return self.vertices[sample_i, :, :, frame_i].squeeze().tolist()

    def get_trimesh(self, sample_i, frame_i, offset=None):
        return Trimesh(vertices=self.get_vertices(sample_i, frame_i, offset), faces=self.faces)
    
    def get_traj_sphere(self, mesh):
        root_posi = np.copy(mesh.vertices).mean(0) # (6000, 3)
        root_posi[1]  = self.vertices.numpy().min(axis=(0, 1, 3))[1] + 0.1
        mesh = trimesh.primitives.Sphere(radius=0.05, center=root_posi, transform=None, subdivisions=1)
        return mesh

    def save_obj(self, save_path, frame_i, offset=None):
        mesh = self.get_trimesh(0, frame_i, offset)
        ground_sph_mesh = self.get_traj_sphere(mesh)
        loc_obj_name = os.path.splitext(os.path.basename(save_path))[0] + "_ground_loc.obj"
        ground_save_path = os.path.join(os.path.dirname(save_path), "loc", loc_obj_name)
        with open(save_path, 'w') as fw:
            mesh.export(fw, 'obj')
        # with open(ground_save_path, 'w') as fw:
        #     ground_sph_mesh.export(fw, 'obj')
        return save_path     
    
    def interpolate_motions(self, motion_tensor, interpolate):
        root_loc = motion_tensor[:, -1, :6, :].reshape(1, 1, 6, -1)
        thetas = motion_tensor[:, :24, :6, :].reshape(1, 24, 6, -1)
        
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
            
        
    
    