import pickle
import os
import numpy as np
import torch
from visualize.format_sequences import format_joint_sequences
from visualize.vis_utils import npy2obj
from visualize.interpolate import interpolate
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
        
    p1_jnts = data.get('full_refine_pred_p1_15fps_jnts_list')
    if p1_jnts is None:
        p1_jnts = data.get('full_refine_pseudo_gt_p1_15fps_jnts_list')
            
    p2_jnts = data.get('full_refine_pred_p2_15fps_jnts_list') 
    if p2_jnts is None:
        p2_jnts = data.get('full_refine_pseudo_gt_p2_15fps_jnts_list')
        
    obj_verts_list = data['filtered_obj_verts_list']
    obj_faces_list = data['obj_faces_list']
    
    # Convert joint sequences to numpy if they are torch tensors
    if isinstance(p1_jnts, torch.Tensor):
        p1_jnts = p1_jnts.detach().cpu().numpy()
    if isinstance(p2_jnts, torch.Tensor):
        p2_jnts = p2_jnts.detach().cpu().numpy()
    
    return p1_jnts, p2_jnts, obj_verts_list, obj_faces_list

def setup_directories(data_file):
    """Create necessary output directories"""
    output_dir = os.path.join("output", os.path.splitext(os.path.basename(data_file))[0])
    p1_dir = os.path.join(output_dir, "p1")
    p2_dir = os.path.join(output_dir, "p2") 
    obj_dir = os.path.join(output_dir, "obj")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(p1_dir, exist_ok=True)
    os.makedirs(p2_dir, exist_ok=True)
    os.makedirs(obj_dir, exist_ok=True)
    
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
        converter1 = npy2obj(data_dict, sample_idx=0, rep_idx=0, device=0, interpolate=2.0, cuda=True)
        converter2 = npy2obj(data_dict, sample_idx=0, rep_idx=1, device=0, interpolate=2.0, cuda=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        with open(cache_file, 'wb') as f:
            pickle.dump((converter1, converter2), f)
            
    return converter1, converter2

def save_obj_files(output_dir, p1_dir, p2_dir, obj_dir, converter1, converter2, converter_obj):
    """Save obj files for each frame"""
    # Save root locations
    root_loc1 = converter1.get_traj()
    root_loc2 = converter2.get_traj()
    
    root_locs = {
        'root_loc1': root_loc1,
        'root_loc2': root_loc2
    }
    np.save(os.path.join(output_dir, "root_locs.npy"), root_locs)
    
    num_frames = min(converter_obj.num_frames, converter1.num_frames, converter2.num_frames)
    
    for frame_i in range(num_frames):
        obj_path = os.path.join(p1_dir, f"frame_{frame_i:04d}.obj")
        converter1.save_obj(obj_path, frame_i)
        
        obj_path = os.path.join(p2_dir, f"frame_{frame_i:04d}.obj")
        converter2.save_obj(obj_path, frame_i)
        
        obj_path = os.path.join(obj_dir, f"frame_{frame_i:04d}.obj")
        converter_obj.save_frame_obj(obj_path,frame_i)
        
        # Print progress bar
        progress = (frame_i + 1) / num_frames * 100
        print(f"\rSaving frames: [{('=' * int(progress/2)).ljust(50)}] {progress:.1f}%", end='', flush=True)
    print() # New line after progress bar completes
        
    

def main():
    args = parse_args()
    data_file = args.data_file

    # Load data
    p1_jnts, p2_jnts, obj_verts_list, obj_faces_list = load_data(data_file)
    
    # Format sequences
    data_dict = format_joint_sequences(p1_jnts, p2_jnts)
    # Setup directories
    output_dir, p1_dir, p2_dir, obj_dir = setup_directories(data_file)
    
    # Get converters
    converter1, converter2 = get_converters(data_dict, data_file)
    # Setup interpolator
    converter_obj = interpolate(obj_verts_list, obj_faces_list, interpolate=2.0)
    
    save_obj_files(output_dir, p1_dir, p2_dir, obj_dir, converter1, converter2, converter_obj)

if __name__ == "__main__":
    main()