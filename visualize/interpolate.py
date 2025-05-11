import numpy as np
from trimesh import Trimesh

class interpolate:
    def __init__(self, verts_list, faces_list, interpolate):
        self.verts_list, self.num_frames = self.interpolate_verts_list(verts_list, interpolate)
        self.faces_list = faces_list

    def interpolate_verts_list(self, verts_list, interpolate):
        """
        Interpolate vertex positions between frames to increase frame count.
        
        Args:
            verts_list: numpy array of shape (n_frames, n_verts, 3) containing vertex positions
            interpolate: float multiplier for number of frames (e.g. 2.0 doubles frame count)
        
        Returns:
            Interpolated vertex positions array of shape (n_frames_new, n_verts, 3)
        """
        n_frames, n_verts, _ = verts_list.shape
        total_frames = int((n_frames - 1) * interpolate + 1)
        
        interp_verts = np.zeros((total_frames, n_verts, 3))
        
        for k in range(total_frames):
            frame_pos = k / interpolate
            frame_idx = int(frame_pos)
            alpha = frame_pos - frame_idx
            
            if frame_idx >= n_frames - 1:
                # Handle last frame
                interp_verts[k] = verts_list[-1]
            else:
                # Linear interpolation between adjacent frames
                v1 = verts_list[frame_idx]
                v2 = verts_list[frame_idx + 1]
                interp_verts[k] = v1 + alpha * (v2 - v1)
                
        return interp_verts, total_frames

    def save_frame_obj(self, save_path, frame_idx):
        """
        Save a single frame's vertices and faces as an obj file.
        
        Args:

            frame_idx: int, index of frame to save
            save_path: str, path to save the obj file
        """
        # Get vertices for this frame
        vertices = self.verts_list[frame_idx]
        
        # Create trimesh object
        mesh = Trimesh(vertices=vertices, faces=self.faces_list)

        # Export to obj file
        with open(save_path, 'w') as f:
            mesh.export(f, 'obj')

        return save_path
