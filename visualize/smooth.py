import os
import pickle
import argparse
import torch
import matplotlib.pyplot as plt
import visualize.utils.rotation_conversions as geometry

def slerp(R1, R2, alpha):
    R1_inv = R1.transpose(-2, -1)
    R_diff = torch.matmul(R1_inv, R2)
    axis_angle = geometry.matrix_to_axis_angle(R_diff)
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / (angle + 1e-7)
    scaled_angle = angle * alpha
    scaled_axis_angle = axis * scaled_angle
    R_interp = geometry.axis_angle_to_matrix(scaled_axis_angle)
    return torch.matmul(R1, R_interp)

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

def calculate_joint_accelerations(thetas):
    n_frames = thetas.shape[-1]
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
        
        # Pad with zeros at start and end
        accels = torch.cat(
            [torch.zeros(1).to(accels.device),
             accels,
             torch.zeros(1).to(accels.device)])
        accelerations.append(accels)

    return torch.stack(accelerations) # [24, n_frames]

def get_jerk_intervals(accelerations, thereshold = 1.0, expand_frames = 1):
    n_frames = accelerations.shape[1]
    intervals = []
    start_idx = None
    last_idx = None
    frames_since_last = 0
    
    # Find intervals in a single pass
    for i in range(n_frames):
        # Check if any joint exceeds threshold
        if torch.any(torch.abs(accelerations[:,i]) > thereshold):
            if start_idx is None:
                start_idx = i
            frames_since_last = 0
            last_idx = i
        elif frames_since_last > expand_frames and start_idx is not None:
            intervals.append([start_idx, last_idx])
            start_idx = None
        frames_since_last += 1
            
    # Handle final interval if exists
    if start_idx is not None:
        intervals.append([start_idx, last_idx])
        
    return intervals

def smooth_motion(motion_tensor):
    thetas = motion_tensor[:, :-1] # [1, 24, 9, n]
    _, n_joints, _, n_frames = thetas.shape
    
    thetas = thetas.reshape(1, n_joints, 3, 3, n_frames)
    accelerations = calculate_joint_accelerations(thetas)
    intervals = get_jerk_intervals(accelerations)
    
    smoothed_motion = thetas.clone()
    
    for interval in intervals:
        start, end = interval
        # prevent out of bounds, but not gonna happen
        if start == 0: start += 1
        if end == n_frames - 1: end -= 1
        
        for i in range(start, end+1):
            alpha = (i - start + 1) / (end - start + 2)
            # smoothed_motion[0, :, :, :, i] = thetas[0, :, :, :, start-1]
            smoothed_motion[0, :, :, :, i] = slerp(thetas[0, :, :, :, start-1], thetas[0, :, :, :, end+1], alpha)
    
    smoothed_motion = smoothed_motion.reshape(1, n_joints, 9, n_frames)
    smoothed_motion = torch.cat([smoothed_motion, motion_tensor[:,-1:]], dim=1)  # [1, 25, 9, n]
    
    return smoothed_motion

def main():
    args = parse_args()
    motion_tensors = load_motion_tensors(args.data_file)
    
    angles_list = []
    intervals_list = []
    for tensor in motion_tensors:
        smoothed_motion = smooth_motion(tensor)
        print(smoothed_motion.shape)
        
if __name__ == "__main__":
    main()
