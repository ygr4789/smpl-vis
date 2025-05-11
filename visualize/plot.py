
import matplotlib.pyplot as plt
import numpy as np

import argparse
import pickle

def plot_joints(p1_joints, p2_joints, obj_verts_list):
    print(f"p1_joints shape: {p1_joints.shape}")
    print(f"p2_joints shape: {p2_joints.shape}")
    print(f"obj_verts_list shape: {obj_verts_list.shape}")
    
    # Create figure and axes
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create slider axis
    slider_ax = plt.axes([0.2, 0.02, 0.6, 0.03])
    frame_slider = plt.Slider(slider_ax, 'Frame', 0, p1_joints.shape[0]-1, 
                            valinit=0, valstep=1)
    
    scatter_p1 = None
    scatter_p2 = None
    scatter_obj = None
    
    def update(val):
        nonlocal scatter_p1, scatter_p2, scatter_obj
        
        # Get current frame from slider
        frame = int(frame_slider.val)
        
        # Clear previous scatter plots
        if scatter_p1 is not None:
            scatter_p1.remove()
        if scatter_p2 is not None:
            scatter_p2.remove()
        if scatter_obj is not None:
            scatter_obj.remove()
            
        # Plot joints for p1 at selected frame
        scatter_p1_points = []
        for joint_idx in range(p1_joints.shape[1]):
            joint_pos = p1_joints[frame, joint_idx]
            scatter_p1_points.append([joint_pos[0], joint_pos[2], joint_pos[1]])
        scatter_p1_points = np.array(scatter_p1_points)
        scatter_p1 = ax.scatter(scatter_p1_points[:,0], scatter_p1_points[:,1], scatter_p1_points[:,2],
                              c='red', s=50, label='p1 joints')
            
        # Plot joints for p2 at selected frame
        scatter_p2_points = []  
        for joint_idx in range(p2_joints.shape[1]):
            joint_pos = p2_joints[frame, joint_idx]
            scatter_p2_points.append([joint_pos[0], joint_pos[2], joint_pos[1]])
        scatter_p2_points = np.array(scatter_p2_points)
        scatter_p2 = ax.scatter(scatter_p2_points[:,0], scatter_p2_points[:,1], scatter_p2_points[:,2],
                              c='blue', s=50, label='p2 joints')
        
        # Plot obj at selected frame
        scatter_obj_points = []
        for vert_idx in range(obj_verts_list.shape[1]):
            vert_pos = obj_verts_list[frame, vert_idx]
            scatter_obj_points.append([vert_pos[0], vert_pos[2], vert_pos[1]])
        scatter_obj_points = np.array(scatter_obj_points)
        scatter_obj = ax.scatter(scatter_obj_points[:,0], scatter_obj_points[:,1], scatter_obj_points[:,2],
                              c='green', s=50, label='obj verts')
        
        # Update title
        ax.set_title(f'Joint Positions at Frame {frame}')
        
        # Redraw
        fig.canvas.draw_idle()
    
    # Connect slider to update function
    frame_slider.on_changed(update)
    
    # Initial plot
    update(0)
    
    # Get axis limits
    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()
    z_lims = ax.get_zlim()
    
    # Find maximum range
    max_range = max([x_lims[1]-x_lims[0], y_lims[1]-y_lims[0], z_lims[1]-z_lims[0]])
    
    # Get centers
    x_mid = (x_lims[1] + x_lims[0])/2
    y_mid = (y_lims[1] + y_lims[0])/2
    z_mid = (z_lims[1] + z_lims[0])/2
    
    # Set equal aspect ratio
    ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
    ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
    ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)
    
    # Labels
    ax.set_xlabel('X Position')
    ax.set_ylabel('Z Position')
    ax.set_zlabel('Y Position')
    ax.legend()
    ax.set_box_aspect([1,1,1])
    
    plt.show()
    
def plot_trajectories(points_trajs):
    """Plot trajectories of points over time.
    
    Args:
        points_trajs: Array of shape [frames, num_points, 3] or [frames, 3] containing point trajectories
    """
    points_trajs = np.array(points_trajs)
    if len(points_trajs.shape) == 2:
        points_trajs = points_trajs.reshape(points_trajs.shape[0], 1, points_trajs.shape[1])
    
    num_frames, num_points, _ = points_trajs.shape
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory for each point
    colors = plt.cm.rainbow(np.linspace(0, 1, num_points))
    for point_idx in range(num_points):
        trajectory = points_trajs[:, point_idx]
        ax.plot(trajectory[:, 0], trajectory[:, 2], trajectory[:, 1], 
                c=colors[point_idx], label=f'Point {point_idx}')
        
        # Plot start and end points
        ax.scatter(trajectory[0, 0], trajectory[0, 2], trajectory[0, 1], 
                  c=colors[point_idx], marker='o', s=100)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 2], trajectory[-1, 1],
                  c=colors[point_idx], marker='s', s=100)
    
    # Get axis limits
    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim() 
    z_lims = ax.get_zlim()
    
    # Find maximum range
    max_range = max([x_lims[1]-x_lims[0], y_lims[1]-y_lims[0], z_lims[1]-z_lims[0]])
    
    # Get centers
    x_mid = (x_lims[1] + x_lims[0])/2
    y_mid = (y_lims[1] + y_lims[0])/2
    z_mid = (z_lims[1] + z_lims[0])/2
    
    # Set equal aspect ratio
    ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
    ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
    ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)
    
    # Labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Z Position')
    ax.set_zlabel('Y Position')
    ax.set_title('Point Trajectories')
    ax.legend()
    ax.set_box_aspect([1,1,1])
    
    plt.show()
    
def main():
    parser = argparse.ArgumentParser(description='Process motion data file')
    parser.add_argument('data_file', type=str, help='Path to the data file')
    args = parser.parse_args()
    
    with open(args.data_file, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        
    p1_jnts = data.get('full_refine_pred_p1_20fps_jnts_list')
    p2_jnts = data.get('full_refine_pred_p2_20fps_jnts_list')
    obj_verts_list = data['filtered_obj_verts_list']

    plot_joints(p1_jnts, p2_jnts, obj_verts_list)
    # plot_trajectories(obj_verts_list)
    
if __name__ == "__main__":
    main()