import numpy as np
from trimesh import Trimesh

from visualize.converter import converter

class converter_vf2obj(converter):
    def __init__(self, verts_list, faces_list, interpolate):
        self.verts_list = verts_list
        self.faces_list = faces_list
        self.interpolate = interpolate
        self.original_num_frames = verts_list.shape[0]
        self.num_frames = int(self.original_num_frames * interpolate)

    def get_interpolated_vertices(self, frame_idx):
        frame_pos = frame_idx / self.interpolate
        orig_frame_idx = int(frame_pos)
        alpha = frame_pos - orig_frame_idx
        
        if orig_frame_idx >= self.original_num_frames - 1:
            # Handle last frame
            return self.verts_list[-1]
        else:
            # Linear interpolation between adjacent frames
            v1 = self.verts_list[orig_frame_idx]
            v2 = self.verts_list[orig_frame_idx + 1]
            return v1 + alpha * (v2 - v1)

    def save_obj(self, save_path, frame_idx):
        # Get interpolated vertices for this frame
        vertices = self.get_interpolated_vertices(frame_idx)
        
        # Create trimesh object
        mesh = Trimesh(vertices=vertices, faces=self.faces_list)

        # Export to obj file
        with open(save_path, 'w') as f:
            mesh.export(f, 'obj')

        return save_path