import pickle
import os
import numpy as np
from visualize.format_sequences import format_joint_sequences
from visualize.vis_utils import npy2obj
from visualize.interpolate import interpolate

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Process motion data file')
    parser.add_argument('data_file', type=str, help='Path to the data file')
    return parser.parse_args()

args = parse_args()
data_file = args.data_file

# Load the pickle file
with open(data_file, 'rb') as f:
# with open('SMPL/mano/MANO_LEFT.pkl', 'rb') as f:
    # Try loading with Python 2 compatibility
    data = pickle.load(f, encoding='latin1')
    
print("Data keys:")
for key in data.keys():
    print(f"- {key}")
    
p1_jnts = data['full_refine_pred_p1_20fps_jnts_list']
p2_jnts = data['full_refine_pred_p2_20fps_jnts_list']
obj_verts_list = data['filtered_obj_verts_list']
obj_faces_list = data['obj_faces_list']

# Print all keys in the data dictionary
p1_trans = data['pred_p1_trans_list']
p2_trans = data['pred_p2_trans_list']

print("Original shapes:")
print(f"p1_jnts: {p1_jnts.shape}")
print(f"p2_jnts: {p2_jnts.shape}")
print(f"p1_trans: {p1_trans.shape}")
print(f"p2_trans: {p2_trans.shape}")
print(f"obj_verts_list: {obj_verts_list.shape}")
print(f"obj_faces_list: {obj_faces_list.shape}")

# Format and save the sequences
data_dict = format_joint_sequences(p1_jnts, p2_jnts)

# Convert to obj files
output_dir = "obj_output"
p1_dir = "p1"
p2_dir = "p2"
obj_dir = "obj"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, p1_dir), exist_ok=True)
os.makedirs(os.path.join(output_dir, p2_dir), exist_ok=True)
os.makedirs(os.path.join(output_dir, obj_dir), exist_ok=True)

# Load the npy file and convert to obj
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
    
    # Save both converters in one file
    with open(cache_file, 'wb') as f:
        pickle.dump((converter1, converter2), f)

interpolator = interpolate(obj_verts_list, obj_faces_list, interpolate=2.0)

num_frames = converter1.num_frames

conv1_root_pos = converter1.root_loc
conv2_root_pos = converter2.root_loc

# Plot trajectories
import matplotlib.pyplot as plt

# Extract coordinates for plotting
conv1_pos = conv1_root_pos[0, 0].T  # Shape: (n_frames, 3)
conv2_pos = conv2_root_pos[0, 0].T  # Shape: (n_frames, 3)

p1_pos = p1_trans[:-30].cpu().numpy()  # Shape: (n_frames, 3) 
p2_pos = p2_trans[:-30].cpu().numpy()  # Shape: (n_frames, 3)

# Interpolate trajectories to match frame counts
min_frames = min(len(conv1_pos), len(p1_pos))
t_target = np.linspace(0, 1, min_frames)
t_source = np.linspace(0, 1, len(conv1_pos))

conv1_interp = np.array([np.interp(t_target, t_source, conv1_pos[:, i]) for i in range(3)]).T
conv2_interp = np.array([np.interp(t_target, t_source, conv2_pos[:, i]) for i in range(3)]).T

t_source = np.linspace(0, 1, len(p1_pos))
p1_interp = np.array([np.interp(t_target, t_source, p1_pos[:, i]) for i in range(3)]).T
p2_interp = np.array([np.interp(t_target, t_source, p2_pos[:, i]) for i in range(3)]).T

# Calculate translation offsets
trans_offset1 = np.mean(p1_interp - conv1_interp, axis=0)
trans_offset2 = np.mean(p2_interp - conv2_interp, axis=0)

# Apply offsets
conv1_aligned = conv1_pos + trans_offset1
conv2_aligned = conv2_pos + trans_offset2

# Plot aligned trajectory
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.plot(conv1_pos[:, 0], conv1_pos[:, 2], conv1_pos[:, 1], 'b-', label='SMPL Root Position 1')
ax.plot(conv2_pos[:, 0], conv2_pos[:, 2], conv2_pos[:, 1], 'b--', label='SMPL Root Position 2')
ax.plot(p1_pos[:, 0], p1_pos[:, 2], p1_pos[:, 1], 'r-', label='Original Translation 1')
ax.plot(p2_pos[:, 0], p2_pos[:, 2], p2_pos[:, 1], 'r--', label='Original Translation 2')
ax.plot(conv1_pos[:, 0] + trans_offset1[0], conv1_pos[:, 2] + trans_offset1[2], conv1_pos[:, 1] + trans_offset1[1], 'g-', label='Aligned SMPL Root Position 1')
ax.plot(conv2_pos[:, 0] + trans_offset2[0], conv2_pos[:, 2] + trans_offset2[2], conv2_pos[:, 1] + trans_offset2[1], 'g--', label='Aligned SMPL Root Position 2')

ax.set_xlabel('X Position')
ax.set_ylabel('Z Position') 
ax.set_zlabel('Y Position')
ax.set_title('3D View of Trajectories')
ax.legend()
ax.set_box_aspect([1,1,1])

print(trans_offset1)
print(trans_offset2)

plt.show()
exit()

# Convert each frame to obj
for frame_i in range(num_frames):
    obj_path = os.path.join(output_dir, p1_dir, f"frame_{frame_i:04d}.obj")
    converter1.save_obj(obj_path, frame_i, offset=trans_offset1)
    # converter1.save_obj(obj_path, frame_i)
    obj_path = os.path.join(output_dir, p2_dir, f"frame_{frame_i:04d}.obj")
    converter2.save_obj(obj_path, frame_i, offset=trans_offset2)
    # converter2.save_obj(obj_path, frame_i)
    obj_path = os.path.join(output_dir, obj_dir, f"frame_{frame_i:04d}.obj")
    interpolator.save_frame_obj(frame_i, obj_path)
    print(f"Saved frame {frame_i} to {obj_path}")