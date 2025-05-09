import pickle
import torch
import numpy as np
import os
from visualize.format_sequences import format_joint_sequences
from visualize.vis_utils import npy2obj

# Load the pickle file
with open('data/sample_41_seq_0_test.pkl', 'rb') as f:
# with open('SMPL/mano/MANO_LEFT.pkl', 'rb') as f:
    # Try loading with Python 2 compatibility
    data = pickle.load(f, encoding='latin1')
    
p1_jnts = data['full_refine_pred_p1_20fps_jnts_list']
p2_jnts = data['full_refine_pred_p2_20fps_jnts_list']

print("Original shapes:")
print(f"p1_jnts: {p1_jnts.shape}")
print(f"p2_jnts: {p2_jnts.shape}")

# Format and save the sequences
data_dict = format_joint_sequences(p1_jnts, p2_jnts)
output_path = 'formatted_sequences.npy'
np.save(output_path, data_dict)
print(f"Saved formatted sequences to {output_path}")

# Convert to obj files
output_dir = "obj_output"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "loc"), exist_ok=True)

# Load the npy file and convert to obj
converter = npy2obj(output_path, sample_idx=0, rep_idx=0, device=0, cuda=True)
num_frames = converter.num_frames

# Convert each frame to obj
for frame_i in range(num_frames):
    obj_path = os.path.join(output_dir, f"frame_{frame_i:04d}.obj")
    converter.save_obj(obj_path, frame_i)
    print(f"Saved frame {frame_i} to {obj_path}")

