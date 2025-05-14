import pickle
import os
import numpy as np
import torch

from visualize.format_sequences import format_joint_sequences
from visualize.converter_rot2obj import converter_rot2obj
from visualize.converter_vf2obj import converter_vf2obj
from visualize.jnt2rot_wrapper import jnt2rot_wrapper
from blender.const import *

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Process motion data file')
    parser.add_argument('data_file', type=str, help='Path to the data file')
    return parser.parse_args()


def load_data(data_file):
    with open(data_file, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        
    p1_jnts_input = data[KEY_INPUT_P1_JNTS]
    p2_jnts_input = data[KEY_INPUT_P2_JNTS]
    obj_verts_list_original = data[KEY_ORIGINAL_OBJ_VERTS]
    p1_jnts_refine = data[KEY_REFINE_P1_JNTS]
    p2_jnts_refine = data[KEY_REFINE_P2_JNTS]
    obj_verts_list_filtered = data[KEY_FILTERED_OBJ_VERTS]
        
    p1_jnts_input = p1_jnts_input.numpy() if torch.is_tensor(p1_jnts_input) else p1_jnts_input
    p2_jnts_input = p2_jnts_input.numpy() if torch.is_tensor(p2_jnts_input) else p2_jnts_input
    obj_verts_list_original = obj_verts_list_original.numpy() if torch.is_tensor(obj_verts_list_original) else obj_verts_list_original
    p1_jnts_refine = p1_jnts_refine.numpy() if torch.is_tensor(p1_jnts_refine) else p1_jnts_refine
    p2_jnts_refine = p2_jnts_refine.numpy() if torch.is_tensor(p2_jnts_refine) else p2_jnts_refine
    obj_verts_list_filtered = obj_verts_list_filtered.numpy() if torch.is_tensor(obj_verts_list_filtered) else obj_verts_list_filtered
    
    obj_faces_list = data[KEY_OBJ_FACES]
    data_type = data[KEY_TYPE]
    cam_T = data[KEY_CAM_T]
    
    return p1_jnts_input, p2_jnts_input, obj_verts_list_original, p1_jnts_refine, p2_jnts_refine, obj_verts_list_filtered, obj_faces_list, data_type, cam_T


def setup_directories(data_file):
    output_dir = os.path.join("output", os.path.splitext(os.path.basename(data_file))[0])
    
    p1_input_dir = os.path.join(output_dir, OBJ_P1_INPUT)
    p2_input_dir = os.path.join(output_dir, OBJ_P2_INPUT) 
    obj_original_dir = os.path.join(output_dir, OBJ_OBJ_ORIGINAL)
    p1_refine_dir = os.path.join(output_dir, OBJ_P1_REFINE)
    p2_refine_dir = os.path.join(output_dir, OBJ_P2_REFINE)
    obj_filtered_dir = os.path.join(output_dir, OBJ_OBJ_FILTERED)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(p1_input_dir, exist_ok=True)
    os.makedirs(p2_input_dir, exist_ok=True)
    os.makedirs(obj_original_dir, exist_ok=True)
    os.makedirs(p1_refine_dir, exist_ok=True)
    os.makedirs(p2_refine_dir, exist_ok=True)
    os.makedirs(obj_filtered_dir, exist_ok=True)
    
    return output_dir, p1_input_dir, p2_input_dir, obj_original_dir, p1_refine_dir, p2_refine_dir, obj_filtered_dir


def get_converters(data_dict, data_file):
    cache_dir = CACHE_DIR
    cache_file = os.path.join(cache_dir, data_file.split('/')[-1].split('.')[0] + CACHE_SUFFIX)

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            converters = pickle.load(f)
            motion_tensor_1_input, motion_tensor_2_input, motion_tensor_1_refine, motion_tensor_2_refine = converters
    else:
        motion_tensor_1_input = jnt2rot_wrapper(data_dict, sample_idx=0, device=0, cuda=True).get_motion_tensor()
        motion_tensor_2_input = jnt2rot_wrapper(data_dict, sample_idx=1, device=0, cuda=True).get_motion_tensor()
        motion_tensor_1_refine = jnt2rot_wrapper(data_dict, sample_idx=2, device=0, cuda=True).get_motion_tensor()
        motion_tensor_2_refine = jnt2rot_wrapper(data_dict, sample_idx=3, device=0, cuda=True).get_motion_tensor()
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump((motion_tensor_1_input, motion_tensor_2_input, motion_tensor_1_refine, motion_tensor_2_refine), f)
        
    converter1_input = converter_rot2obj(motion_tensor_1_input, interpolate=INTERPOLATE, device=0, cuda=True)
    converter2_input = converter_rot2obj(motion_tensor_2_input, interpolate=INTERPOLATE, device=0, cuda=True)
    converter1_refine = converter_rot2obj(motion_tensor_1_refine, interpolate=INTERPOLATE, device=0, cuda=True)
    converter2_refine = converter_rot2obj(motion_tensor_2_refine, interpolate=INTERPOLATE, device=0, cuda=True)
            
    return converter1_input, converter2_input, converter1_refine, converter2_refine


def save_obj_files(dirs, converters):
    num_frames = min([converter.num_frames for converter in converters])
    for frame_i in range(num_frames):
        for dir_path, converter in zip(dirs, converters):
            obj_path = os.path.join(dir_path, f"frame_{frame_i:04d}.obj")
            converter.save_obj(obj_path, frame_i)
        
        progress = (frame_i + 1) / num_frames * 100
        print(f"\rSaving frames: [{('=' * int(progress/2)).ljust(50)}] {progress:.1f}%", end='', flush=True)
    print() # New line after progress bar completes


def save_info(output_dir, data_type, root_loc1, root_loc2, cam_T):
    info = {
        INFO_ROOT_LOC_P1: root_loc1,
        INFO_ROOT_LOC_P2: root_loc2,
        INFO_TYPE: data_type,
        INFO_CAM: cam_T
    }
    np.save(os.path.join(output_dir, INFO_FILE_NAME), info)


def main():
    args = parse_args()
    data_file = args.data_file

    # Load data
    p1_jnts_input, p2_jnts_input, obj_verts_list_original, p1_jnts_refine, p2_jnts_refine, obj_verts_list_filtered, obj_faces_list, data_type, cam_T = load_data(data_file)
    
    # Format sequences
    data_dict = format_joint_sequences(p1_jnts_input, p2_jnts_input, p1_jnts_refine, p2_jnts_refine)
    
    # Setup Converters
    converter1_input, converter2_input, converter1_refine, converter2_refine = get_converters(data_dict, data_file)
    converter_obj_original = converter_vf2obj(obj_verts_list_original, obj_faces_list, interpolate=INTERPOLATE)
    converter_obj_filtered = converter_vf2obj(obj_verts_list_filtered, obj_faces_list, interpolate=INTERPOLATE)
    converters = [converter1_input, converter2_input, converter1_refine, converter2_refine, converter_obj_original, converter_obj_filtered]
    
    # Setup directories
    output_dir, p1_input_dir, p2_input_dir, obj_original_dir, p1_refine_dir, p2_refine_dir, obj_filtered_dir = setup_directories(data_file)
    dirs = [p1_input_dir, p2_input_dir, p1_refine_dir, p2_refine_dir, obj_original_dir, obj_filtered_dir]
    
    save_obj_files(dirs, converters)
    save_info(output_dir, data_type, converter1_input.get_traj(), converter2_input.get_traj(), cam_T)

if __name__ == "__main__":
    main()