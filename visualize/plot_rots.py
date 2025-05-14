import os
import pickle
import argparse
import torch
import utils.rotation_conversions as geometry
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_file', type=str, required=True)
    return parser.parse_args()

def load_motion_tensors(data_file):
    with open(data_file, 'rb') as f:
        data_dict = pickle.load(f)
        motion_tensor_1_input, motion_tensor_2_input, motion_tensor_1_refine, motion_tensor_2_refine = data_dict
    return (motion_tensor_1_input.cpu(), motion_tensor_2_input.cpu(), 
            motion_tensor_1_refine.cpu(), motion_tensor_2_refine.cpu())

def calculate_joint_angles(motion_tensor):
    thetas = motion_tensor[:, :-1] # [1, 24, 9, n]
    root_loc = motion_tensor[:, -1] # [1, 1, 9, n]

    # Reshape thetas to [1, 24, 3, 3, n] for rotation matrix form
    n_frames = thetas.shape[-1]
    thetas = thetas.reshape(1, 24, 3, 3, n_frames)

    # Calculate angle differences for all joints
    accelerations = []
    for joint_idx in range(24):
        joint_rots = thetas[0, joint_idx] # [3, 3, n_frames]
        joint_rots = joint_rots.permute(2, 0, 1) # [n_frames, 3, 3]
        
        # Calculate rotation differences R1^-1R2, R2^-1R3, etc
        rot_diffs = []
        for i in range(n_frames-1):
            R1 = joint_rots[i]
            R2 = joint_rots[i+1]
            R1_inv = R1.transpose(0,1)
            diff = torch.matmul(R1_inv, R2)
            rot_diffs.append(diff)
            
        rot_diffs = torch.stack(rot_diffs) # [n_frames-1, 3, 3]
        
        # Convert to axis-angle and get magnitudes (velocities)
        axis_angles = geometry.matrix_to_axis_angle(rot_diffs) # [n_frames-1, 3]
        velocities = torch.norm(axis_angles, dim=1) # [n_frames-1]
        
        # Calculate acceleration as difference of velocities
        accels = velocities[1:] - velocities[:-1] # [n_frames-2]
        accelerations.append(accels)

    return torch.stack(accelerations), n_frames-1 # [24, n_frames-2]

def plot_angles(angles_list, n_frames_list, titles):
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    axs = axs.ravel()
    
    for idx, (angles, n_frames, title) in enumerate(zip(angles_list, n_frames_list, titles)):
        time = range(n_frames-1)
        for joint_idx in range(24):
            axs[idx].plot(time, angles[joint_idx], label=f'Joint {joint_idx}', alpha=0.7)
            
        axs[idx].set_xlabel('Frame')
        axs[idx].set_ylabel('Rotation Angle (radians)')
        axs[idx].set_title(title)
        axs[idx].set_ylim(-3, 3)
        axs[idx].grid(True)
        
        # Place legend outside of the last subplot
        if idx == 3:
            axs[idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

def main():
    args = parse_args()
    motion_tensors = load_motion_tensors(args.data_file)
    
    angles_list = []
    n_frames_list = []
    for tensor in motion_tensors:
        angles, n_frames = calculate_joint_angles(tensor)
        angles_list.append(angles)
        n_frames_list.append(n_frames)
        
    titles = ['Person 1 Input', 'Person 2 Input', 
              'Person 1 Refined', 'Person 2 Refined']
              
    plot_angles(angles_list, n_frames_list, titles)

if __name__ == "__main__":
    main()
