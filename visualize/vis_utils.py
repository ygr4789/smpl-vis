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
    def __init__(self, npy_path, sample_idx, rep_idx, device=0, cuda=True):
        self.npy_path = npy_path
        self.motions = np.load(self.npy_path, allow_pickle=True)
        if self.npy_path.endswith('.npz'):
            self.motions = self.motions['arr_0']
        self.motions = self.motions[None][0]
        self.rot2xyz = Rotation2xyz(device='cpu')
        self.faces = self.rot2xyz.smpl_model.faces
        self.bs, self.njoints, self.nfeats, self.nframes = self.motions['motion'].shape
        self.opt_cache = {}
        self.sample_idx = sample_idx
        self.total_num_samples = self.motions['num_samples']
        self.rep_idx = rep_idx
        self.absl_idx = self.sample_idx*self.total_num_samples + self.rep_idx
        self.num_frames = self.motions['motion'][self.absl_idx].shape[-1]
        self.j2s = joints2smpl(num_frames=self.num_frames, device_id=device, cuda=cuda)

        if self.nfeats == 3:
            print(f'Running SMPLify For sample [{sample_idx}], repetition [{rep_idx}], it may take a few minutes.')
            motion_tensor, opt_dict = self.j2s.joint2smpl(self.motions['motion'][self.absl_idx].transpose(2, 0, 1))  # [nframes, njoints, 3]
            # self.motions['motion'] = motion_tensor.cpu().numpy()
            self.motions['motion'] = self.interpolate_motions(motion_tensor).cpu().numpy()
            self.num_frames = self.motions['motion'][self.absl_idx].shape[-1]
            
        elif self.nfeats == 6:
            self.motions['motion'] = self.motions['motion'][[self.absl_idx]]
            
        
        self.bs, self.njoints, self.nfeats, self.nframes = self.motions['motion'].shape
        self.real_num_frames = self.motions['lengths'][self.absl_idx] # 196
        
        self.vertices = self.rot2xyz(torch.tensor(self.motions['motion']), mask=None,
                                     pose_rep='rot6d', translation=True, glob=True,
                                     jointstype='vertices',
                                     # jointstype='smpl',  # for joint locations
                                     vertstrans=True)
        self.root_loc = self.motions['motion'][:, -1, :3, :].reshape(1, 1, 3, -1)

    def get_vertices(self, sample_i, frame_i):
        return self.vertices[sample_i, :, :, frame_i].squeeze().tolist()

    def get_trimesh(self, sample_i, frame_i):
        return Trimesh(vertices=self.get_vertices(sample_i, frame_i),
                       faces=self.faces)
    
    def get_traj_sphere(self, mesh):
        root_posi = np.copy(mesh.vertices).mean(0) # (6000, 3)
        root_posi[1]  = self.vertices.numpy().min(axis=(0, 1, 3))[1] + 0.1
        mesh = trimesh.primitives.Sphere(radius=0.05, center=root_posi, transform=None, subdivisions=1)
        return mesh

    def save_obj(self, save_path, frame_i):
        mesh = self.get_trimesh(0, frame_i)
        ground_sph_mesh = self.get_traj_sphere(mesh)
        loc_obj_name = os.path.splitext(os.path.basename(save_path))[0] + "_ground_loc.obj"
        ground_save_path = os.path.join(os.path.dirname(save_path), "loc", loc_obj_name)
        with open(save_path, 'w') as fw:
            mesh.export(fw, 'obj')
        with open(ground_save_path, 'w') as fw:
            ground_sph_mesh.export(fw, 'obj')
        return save_path
    
    def interpolate_motions(self, motions_tensor):
        # Convert numpy arrays to torch tensors
        root_loc = motions_tensor[:, -1, :6, :].reshape(1, 1, 6, -1)
        thetas = motions_tensor[:, :24, :6, :].reshape(1, 24, 6, -1)
        
        n_frames = thetas.shape[-1]
        
        interp_thetas = torch.zeros((1, 24, 6, 2*n_frames-1), device=thetas.device)
        interp_thetas[..., ::2] = thetas
        
        thetas_reshaped = thetas.permute(0, 3, 1, 2).reshape(-1, 6) # [1, 24, 6, n_frames] to [24*n_frames, 6]
        matrices = geometry.rotation_6d_to_matrix(thetas_reshaped) # [24*n_frames, 3, 3] to [1, 24, 3, 3, n_frames]
        matrices = matrices.reshape(1, n_frames, 24, 3, 3).permute(0, 2, 3, 4, 1)
        
        for i in range(n_frames-1):
            R1 = matrices[..., i]
            R2 = matrices[..., i+1]
            R = geometry.matrix_slerp(R1, R2, 0.5)
            interp = geometry.matrix_to_rotation_6d(R)
            interp_thetas[..., 2*i+1] = interp
            
        interp_root_loc = torch.zeros((1, 1, 6, 2*n_frames-1), device=root_loc.device)
        interp_root_loc[..., ::2] = root_loc
        
        for i in range(n_frames-1):
            p1 = root_loc[..., i]
            p2 = root_loc[..., i+1]
            interp = p1 + 0.5 * (p2 - p1)
            interp_root_loc[..., 2*i+1] = interp
            
        thetas = torch.cat([interp_thetas, interp_root_loc], dim=1)
        
        return thetas  # Convert back to numpy for consistency with the rest of the code
    
        
            
        
    
    