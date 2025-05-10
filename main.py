import pickle
import os
import numpy as np
from visualize.format_sequences import format_joint_sequences
from visualize.vis_utils import npy2obj
from visualize.interpolate import interpolate
from visualize.vis_configs import trans_offset
from visualize.plot import plot_joints, plot_trajectories

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Process motion data file')
    parser.add_argument('data_file', type=str, help='Path to the data file')
    return parser.parse_args()

def load_data(data_file):
    """Load and parse the pickle data file"""
    with open(data_file, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        
    print("Data keys:")
    for key in data.keys():
        print(f"- {key}")
        
    p1_jnts = data['full_refine_pred_p1_20fps_jnts_list']
    p2_jnts = data['full_refine_pred_p2_20fps_jnts_list']
    obj_verts_list = data['filtered_obj_verts_list']
    obj_faces_list = data['obj_faces_list']

    print("Original shapes:")
    print(f"p1_jnts: {p1_jnts.shape}")
    print(f"p2_jnts: {p2_jnts.shape}")
    print(f"obj_verts_list: {obj_verts_list.shape}")
    print(f"obj_faces_list: {obj_faces_list.shape}")

    return p1_jnts, p2_jnts, obj_verts_list, obj_faces_list

def setup_directories():
    """Create necessary output directories"""
    output_dir = "obj_output"
    p1_dir = "p1"
    p2_dir = "p2" 
    obj_dir = "obj"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, p1_dir), exist_ok=True)
    os.makedirs(os.path.join(output_dir, p2_dir), exist_ok=True)
    os.makedirs(os.path.join(output_dir, obj_dir), exist_ok=True)
    
    return output_dir, p1_dir, p2_dir, obj_dir

def get_converters(data_dict, data_file):
    """Get or create npy2obj converters"""
    cache_dir = "cache"
    cache_file = os.path.join(cache_dir, data_file.split('/')[-1].split('.')[0] + "_converters.pkl")

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            converters = pickle.load(f)
            converter1, converter2 = converters
    else:
        converter1 = npy2obj(data_dict, sample_idx=0, rep_idx=0, device=0, interpolate=1.5, cuda=True)
        converter2 = npy2obj(data_dict, sample_idx=0, rep_idx=1, device=0, interpolate=1.5, cuda=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        with open(cache_file, 'wb') as f:
            pickle.dump((converter1, converter2), f)
            
    return converter1, converter2

def save_obj_files(output_dir, p1_dir, p2_dir, obj_dir, num_frames, converter1, converter2, interpolator, trans_offset=None):
    """Save obj files for each frame"""
    for frame_i in range(num_frames):
        obj_path = os.path.join(output_dir, p1_dir, f"frame_{frame_i:04d}.obj")
        converter1.save_obj(obj_path, frame_i, offset=trans_offset)
        
        obj_path = os.path.join(output_dir, p2_dir, f"frame_{frame_i:04d}.obj")
        converter2.save_obj(obj_path, frame_i, offset=trans_offset)
        
        obj_path = os.path.join(output_dir, obj_dir, f"frame_{frame_i:04d}.obj")
        interpolator.save_frame_obj(frame_i, obj_path)
        print(f"Saved frame {frame_i} to {obj_path}")

def main():
    args = parse_args()
    data_file = args.data_file

    # Load data
    p1_jnts, p2_jnts, obj_verts_list, obj_faces_list = load_data(data_file)
    
    # plot_joints(p1_jnts, p2_jnts)
    # return
    
    # Format sequences
    data_dict = format_joint_sequences(p1_jnts, p2_jnts)
    # Setup directories
    output_dir, p1_dir, p2_dir, obj_dir = setup_directories()
    
    # Get converters
    converter1, converter2 = get_converters(data_dict, data_file)
    # Setup interpolator
    interpolator = interpolate(obj_verts_list, obj_faces_list, interpolate=2.0)
    
    save_obj_files(output_dir, p1_dir, p2_dir, obj_dir, converter1.num_frames, converter1, converter2, interpolator)

if __name__ == "__main__":
    main()