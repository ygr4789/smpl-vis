import pickle
import torch
import argparse
import numpy as np
import os

from blender.const import *

def parse_args():
    parser = argparse.ArgumentParser(description='Process motion data file')
    parser.add_argument('input', type=str, help='Path to the data file')
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
        
    # Convert to blender coordinate system (y,z -> z,y and flip y)
    p1_jnts_input[..., [1, 2]] = p1_jnts_input[..., [2, 1]]
    p2_jnts_input[..., [1, 2]] = p2_jnts_input[..., [2, 1]]
    p1_jnts_input[..., 1] *= -1
    p2_jnts_input[..., 1] *= -1
    
    p1_jnts_refine[..., [1, 2]] = p1_jnts_refine[..., [2, 1]]
    p2_jnts_refine[..., [1, 2]] = p2_jnts_refine[..., [2, 1]]
    p1_jnts_refine[..., 1] *= -1
    p2_jnts_refine[..., 1] *= -1
    
    obj_verts_list_original[..., [1, 2]] = obj_verts_list_original[..., [2, 1]]
    obj_verts_list_filtered[..., [1, 2]] = obj_verts_list_filtered[..., [2, 1]]
    obj_verts_list_original[..., 1] *= -1
    obj_verts_list_filtered[..., 1] *= -1
    
    p1_jnts_input = p1_jnts_input.numpy() if torch.is_tensor(p1_jnts_input) else p1_jnts_input
    p2_jnts_input = p2_jnts_input.numpy() if torch.is_tensor(p2_jnts_input) else p2_jnts_input
    obj_verts_list_original = obj_verts_list_original.numpy() if torch.is_tensor(obj_verts_list_original) else obj_verts_list_original
    p1_jnts_refine = p1_jnts_refine.numpy() if torch.is_tensor(p1_jnts_refine) else p1_jnts_refine
    p2_jnts_refine = p2_jnts_refine.numpy() if torch.is_tensor(p2_jnts_refine) else p2_jnts_refine
    obj_verts_list_filtered = obj_verts_list_filtered.numpy() if torch.is_tensor(obj_verts_list_filtered) else obj_verts_list_filtered
    
    obj_faces_list = data[KEY_OBJ_FACES]
    data_type = data[KEY_TYPE]
    
    
    return p1_jnts_input, p2_jnts_input, obj_verts_list_original, p1_jnts_refine, p2_jnts_refine, obj_verts_list_filtered, obj_faces_list, data_type
    
def main():
    args = parse_args()
    input_file = args.input
    
    p1_jnts_input, p2_jnts_input, obj_verts_list_original, p1_jnts_refine, p2_jnts_refine, obj_verts_list_filtered, obj_faces_list, data_type = load_data(input_file)
    
    output_file = os.path.join(CACHE_DIR, os.path.splitext(os.path.basename(input_file))[0] + '.npz')
    # Save data as npz file
    np.savez(output_file,
             **{KEY_INPUT_P1_JNTS: p1_jnts_input,
                KEY_INPUT_P2_JNTS: p2_jnts_input,
                KEY_ORIGINAL_OBJ_VERTS: obj_verts_list_original,
                KEY_REFINE_P1_JNTS: p1_jnts_refine,
                KEY_REFINE_P2_JNTS: p2_jnts_refine,
                KEY_FILTERED_OBJ_VERTS: obj_verts_list_filtered,
                KEY_OBJ_FACES: obj_faces_list,
                KEY_TYPE: data_type})
    
    print(f"Saved data to {output_file}")
    
if __name__ == "__main__":
    main()