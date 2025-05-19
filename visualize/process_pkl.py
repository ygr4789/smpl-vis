import os
import sys
import numpy as np
import pickle
import torch

from visualize.format_sequences import format_joint_sequences
from visualize.converter_rot2obj import converter_rot2obj
from visualize.converter_vf2obj import converter_vf2obj
from visualize.jnt2rot_wrapper import jnt2rot_wrapper
from visualize.const import *


def load_data(data_file, keys_to_process):
    with open(data_file, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    loaded_data = {}
    for key in keys_to_process + [KEY_OBJ_FACES]:
        if key in data:
            value = data[key]
            if torch.is_tensor(value):
                value = value.numpy()
            loaded_data[key] = value
        else:
            raise KeyError(f"Required key '{key}' not found in '{data_file}'")
    
    return loaded_data


def setup_directories(data_file, keys_to_process):
    output_dir = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(data_file))[0])
    os.makedirs(output_dir, exist_ok=True)
    
    dirs = {}
    for key in keys_to_process:
        if key in key_path_map:
            dir_path = os.path.join(output_dir, key_path_map[key])
            os.makedirs(dir_path, exist_ok=True)
            dirs[key] = dir_path
    
    return output_dir, dirs


def get_converters(data_dict, data_file, keys_to_process):
    cache_dir = CACHE_DIR
    cache_file = os.path.join(cache_dir, data_file.split('/')[-1].split('.')[0] + CACHE_SUFFIX)
    
    converters = {}
    
    # Handle joint sequences
    joint_keys = [k for k in keys_to_process if 'jnts' in k]
    if joint_keys:
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                motion_tensors = pickle.load(f)
        else:
            motion_tensors = []
            for i, key in enumerate(joint_keys):
                motion_tensor = jnt2rot_wrapper(data_dict, sample_idx=i, device=0, cuda=True).get_motion_tensor()
                motion_tensors.append(motion_tensor)
            
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(tuple(motion_tensors), f)
        
        for key, motion_tensor in zip(joint_keys, motion_tensors):
            converters[key] = converter_rot2obj(motion_tensor, interpolate=INTERPOLATE, device=0, cuda=True)
    
    obj_keys = [k for k in keys_to_process if 'obj_verts' in k]
    for key in obj_keys:
        if key in data_dict:
            converters[key] = converter_vf2obj(data_dict[key], data_dict[KEY_OBJ_FACES], interpolate=INTERPOLATE)
    
    return converters


def convert_to_blender_coordinates(data):
    data[..., [1, 2]] = data[..., [2, 1]]
    data[..., 1] *= -1
    return data


def save_obj_files(dirs, converters):
    num_frames = min([converter.num_frames for converter in converters.values()])
    for frame_i in range(num_frames):
        for key, converter in converters.items():
            if key in dirs:
                obj_path = os.path.join(dirs[key], f"frame_{frame_i:04d}.obj")
                if not os.path.exists(obj_path):
                    converter.save_obj(obj_path, frame_i)
        
        progress = (frame_i + 1) / num_frames * 100
        print(f"\rSaving obj files: [{('=' * int(progress/2)).ljust(50)}] {progress:.1f}%", end='', flush=True)
    print()


def save_info(output_dir, root_loc1, root_loc2):
    info_path = os.path.join(output_dir, INFO_FILE_NAME)
    if os.path.exists(info_path):
        return
        
    root_loc1 = convert_to_blender_coordinates(root_loc1)
    root_loc2 = convert_to_blender_coordinates(root_loc2)
    
    info = {
        INFO_ROOT_LOC_P1: root_loc1,
        INFO_ROOT_LOC_P2: root_loc2
    }
    np.save(info_path, info)


def process_pkl_file(data_file, keys_to_process=None):
    if keys_to_process is None:
        keys_to_process = [KEY_INPUT_P1_JNTS, KEY_INPUT_P2_JNTS, KEY_ORIGINAL_OBJ_VERTS,
                          KEY_REFINE_P1_JNTS, KEY_REFINE_P2_JNTS, KEY_FILTERED_OBJ_VERTS]
    
    # Load data
    data = load_data(data_file, keys_to_process)
    
    # Format sequences for joint data
    joint_keys = [k for k in keys_to_process if 'jnts' in k]
    data_dict = format_joint_sequences(*[data[k] for k in joint_keys])
    
    # Add all data to data_dict that isn't already present
    for key in data:
        if key not in data_dict:
            data_dict[key] = data[key]
    
    # Setup converters
    print(f"Running SMPLify for {data_file}...")
    converters = get_converters(data_dict, data_file, keys_to_process)
    
    # Setup directories
    output_dir, dirs = setup_directories(data_file, keys_to_process)
    
    # Save obj files
    save_obj_files(dirs, converters)
    
    # Save trajectory info if we have p1/p2 input joints
    p1_keys = [k for k in converters.keys() if 'p1' in k.lower()]
    p2_keys = [k for k in converters.keys() if 'p2' in k.lower()]
    
    if p1_keys and p2_keys:
        save_info(output_dir,
                  converters[p1_keys[0]].get_traj(),
                  converters[p2_keys[0]].get_traj())
    
    # Save data as npz file
    prim_npz_path = os.path.join(output_dir, PRIM_FILE_NAME)
    npz_data = {key: convert_to_blender_coordinates(data[key]) for key in keys_to_process if key in data}
    npz_data[KEY_OBJ_FACES] = data[KEY_OBJ_FACES]
    np.savez(prim_npz_path, **npz_data)
    
    print(f"Done processing for {data_file}.")
    print()