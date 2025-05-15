import bpy
import os
import argparse
import sys
import numpy as np
import threading

from contextlib import contextmanager
from blender.camera import prepare_camera_settings
from blender.const import *
from blender.prim import *
from blender.config import setup_render_settings, setup_animation_settings, stdout_redirected

def parse_arguments():
    # Get all arguments after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    # Create argument parser
    parser = argparse.ArgumentParser(description='Render SMPL visualization in Blender')
    parser.add_argument('obj_folder', type=str, help='Path to pkl data file')
    parser.add_argument('-q', '--high', action='store_true', help='Use high quality rendering settings')
    parser.add_argument('-t', '--target', type=int, choices=[0, 1, 2], 
                       help='Render target: 0=object only, 1=input motion, 2=refined motion', 
                       default=TARGET_FLAG_REFINE)
    parser.add_argument('-c', '--camera', type=int, help='Camera number, -1 for all cameras', default=-1)
    
    return parser.parse_args(argv)

def cleanup_existing_objects():
    """Remove existing mesh objects except Plane"""
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.name != 'Plane':
            bpy.data.objects.remove(obj, do_unlink=True)


def create_mesh_for_frame(verts, faces, frame_num, material):
    """Create mesh object for a specific frame"""
    # Create new mesh datablock
    mesh = bpy.data.meshes.new(f"Frame_{frame_num}_mesh")
    obj = bpy.data.objects.new(f"Frame_{frame_num}", mesh)
    
    # Create mesh from vertices and faces
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    
    # Link object to scene
    bpy.context.scene.collection.objects.link(obj)
    
    # Add material
    obj.data.materials.append(bpy.data.materials[material])
    
    # Smooth shading
    with bpy.context.temp_override(selected_editable_objects=[obj]):
        bpy.ops.object.shade_smooth()
        
    return obj

def setup_keyframes(obj, frame_num):
    """Set up keyframes for visibility of object"""
    # Hide at start
    obj.hide_render = True
    obj.hide_viewport = True
    obj.keyframe_insert(data_path="hide_render", frame=1)
    obj.keyframe_insert(data_path="hide_viewport", frame=1)
    
    # Show at current frame (doubled for interpolation)
    obj.hide_render = False
    obj.hide_viewport = False
    obj.keyframe_insert(data_path="hide_render", frame=frame_num)
    obj.keyframe_insert(data_path="hide_viewport", frame=frame_num)
    
    # Hide at next frame
    obj.hide_render = True
    obj.hide_viewport = True
    obj.keyframe_insert(data_path="hide_render", frame=frame_num + 1)
    obj.keyframe_insert(data_path="hide_viewport", frame=frame_num + 1)

def load_info(obj_folder):
    """Load root locations from numpy file if available"""
    root_locs_path = os.path.join(obj_folder, INFO_FILE_NAME)
    if os.path.exists(root_locs_path):
        info = np.load(root_locs_path, allow_pickle=True).item()
        data_type = info[INFO_TYPE]
        root_loc1 = info[INFO_ROOT_LOC_P1]
        root_loc2 = info[INFO_ROOT_LOC_P2]
        cam_T = info[INFO_CAM]
        return data_type, root_loc1, root_loc2, cam_T
    else:
        raise FileNotFoundError(f"No info file found at: {root_locs_path}")
        
def create_sphere_for_joint(material, joint_idx, radius=0.05):
    """Create sphere object for joint visualization"""
    # Set different radius for each joint
    joint_radius = joint_radii.get(joint_idx, radius)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=joint_radius)
    sphere = bpy.context.active_object
    sphere.data.materials.append(bpy.data.materials[material])
    with bpy.context.temp_override(selected_editable_objects=[sphere]):
        bpy.ops.object.shade_smooth()
    return sphere

def create_bone_cone(joint1_pos, joint2_pos, material, bone_idx, radius=0.03):
    """Create a cone object connecting two joints to represent a bone"""
    # Calculate direction and length
    direction = joint2_pos - joint1_pos
    length = np.linalg.norm(direction)
    direction = direction / length
    # Create cylinder
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=length)
    cone = bpy.context.active_object
    with bpy.context.temp_override(selected_editable_objects=[cone]):
        bpy.ops.object.shade_smooth()
    
    # Add loop cuts to sides
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    # Use subdivide instead of loopcut since it doesn't require view3d context
    bpy.ops.mesh.subdivide(number_cuts=10)
    bpy.ops.object.mode_set(mode='OBJECT')
    # Position and rotate cone
    cone.location = (joint1_pos + joint2_pos) / 2
    
    # Calculate rotation to align cone with direction
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, direction)
    rotation_angle = np.arccos(np.dot(z_axis, direction))
    
    if np.any(rotation_axis):
        cone.rotation_mode = 'AXIS_ANGLE'
        cone.rotation_axis_angle = [rotation_angle] + list(rotation_axis)
    
    # Add material
    cone.data.materials.append(bpy.data.materials[material])
    
    # Smooth shading
    with bpy.context.temp_override(selected_editable_objects=[cone]):
        bpy.ops.object.shade_smooth()
    
    return cone

def load_data(obj_folder):
    """Load and validate data from npz file"""
    npz_file = os.path.join(obj_folder, PRIM_FILE_NAME)
    if not os.path.exists(npz_file):
        raise FileNotFoundError(f"No npz file found at: {npz_file}")
        
    data = np.load(npz_file)
    return data

def create_joints_and_bones(p1_joints, p2_joints):
    """Create joint spheres and bone cones for both characters"""
    # Create spheres for p1 joints
    p1_spheres = [create_sphere_for_joint("Red", i) for i in range(p1_joints.shape[1])]
    # Create spheres for p2 joints  
    p2_spheres = [create_sphere_for_joint("Blue", i) for i in range(p2_joints.shape[1])]
    
    # Create bones for p1
    p1_bones = []
    for bone_idx, (joint1, joint2) in bone_pair.items():
        cone = create_bone_cone(p1_joints[0, joint1], p1_joints[0, joint2], "Red", bone_idx)
        p1_bones.append((cone, joint1, joint2))
        
    # Create bones for p2
    p2_bones = []
    for bone_idx, (joint1, joint2) in bone_pair.items():
        cone = create_bone_cone(p2_joints[0, joint1], p2_joints[0, joint2], "Blue", bone_idx)
        p2_bones.append((cone, joint1, joint2))
        
    return p1_spheres, p2_spheres, p1_bones, p2_bones

def create_object_meshes(verts_list, obj_faces_list):
    """Create mesh objects for each frame"""
    print("Creating object meshes...")
    num_frames = len(verts_list)*2-1
    for frame_num in range(1, num_frames+1):
        # Create mesh for the frame
        if frame_num == num_frames:
            verts = verts_list[-1]
        elif frame_num % 2 == 0:
            verts = verts_list[frame_num//2]
        else:
            verts = (verts_list[frame_num//2] + verts_list[frame_num//2+1]) / 2
        obj = create_mesh_for_frame(verts, obj_faces_list, frame_num, "Yellow")
        setup_keyframes(obj, frame_num)

def update_joints_and_bones(frame_num, p1_joints, p2_joints, p1_spheres, p2_spheres, p1_bones, p2_bones):
    """Update joint and bone positions for a frame"""
    anim_frame = frame_num * 2 - 1
    
    # Update p1 joints and bones
    for sphere, pos in zip(p1_spheres, p1_joints[frame_num-1]):
        sphere.location = pos
        sphere.keyframe_insert(data_path="location", frame=anim_frame)
    
    for cone, joint1, joint2 in p1_bones:
        joint1_pos = p1_joints[frame_num-1, joint1]
        joint2_pos = p1_joints[frame_num-1, joint2]
        update_bone_position(cone, joint1_pos, joint2_pos, anim_frame)
        
    # Update p2 joints and bones
    for sphere, pos in zip(p2_spheres, p2_joints[frame_num-1]):
        sphere.location = pos
        sphere.keyframe_insert(data_path="location", frame=anim_frame)
    
    for cone, joint1, joint2 in p2_bones:
        joint1_pos = p2_joints[frame_num-1, joint1]
        joint2_pos = p2_joints[frame_num-1, joint2]
        update_bone_position(cone, joint1_pos, joint2_pos, anim_frame)

def update_bone_position(cone, joint1_pos, joint2_pos, frame):
    """Update position and rotation of a bone cone"""
    cone.location = (joint1_pos + joint2_pos) / 2
    
    # Update rotation
    direction = joint2_pos - joint1_pos
    length = np.linalg.norm(direction)
    direction = direction / length
    
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, direction)
    rotation_angle = np.arccos(np.dot(z_axis, direction))
    
    if np.any(rotation_axis):
        cone.rotation_mode = 'AXIS_ANGLE'
        cone.rotation_axis_angle = [rotation_angle] + list(rotation_axis)
    
    cone.keyframe_insert(data_path="location", frame=frame)
    cone.keyframe_insert(data_path="rotation_axis_angle", frame=frame)

def render_animation(video_dir, render_target, camera_settings, num_frames):
    """Render animation from different camera angles"""
    for camera_setting in camera_settings:
        video_name = VIDEO_NAMES[render_target] + "_" + camera_setting['text'] + ".mp4"
        os.makedirs(video_dir, exist_ok=True)
        bpy.context.scene.render.filepath = os.path.join(video_dir, video_name)
        bpy.context.scene.camera.location = camera_setting['location']
        bpy.context.scene.camera.rotation_euler = camera_setting['rotation']
        
        print(f"Rendering {num_frames} frames for {camera_setting['text']}...")
        with stdout_redirected(keyword="Fra:", on_match=lambda line: line[:-1].encode()):
            bpy.ops.render.render(animation=True)
        print()
        print(f"Saved to {video_dir} for {camera_setting['text']}")

def main():
    args = parse_arguments()
    obj_folder = args.obj_folder
    render_high = args.high
    render_target = args.target
    camera_no = args.camera
    
    # Load data
    data = load_data(obj_folder)
    p1_jnts_input = data[KEY_INPUT_P1_JNTS]
    p2_jnts_input = data[KEY_INPUT_P2_JNTS]
    obj_verts_list_original = data[KEY_ORIGINAL_OBJ_VERTS]
    p1_jnts_refine = data[KEY_REFINE_P1_JNTS]
    p2_jnts_refine = data[KEY_REFINE_P2_JNTS]
    obj_verts_list_filtered = data[KEY_FILTERED_OBJ_VERTS]
    obj_faces_list = data[KEY_OBJ_FACES]
    data_type = data[KEY_TYPE]
    
    if data_type == TYPE_GT and render_target == TARGET_FLAG_NONE:
        print("Skipping GT data with target flag 0")
        return
    
    # Load scene and setup
    bpy.ops.wm.open_mainfile(filepath=BLENDER_PATH)
    cleanup_existing_objects()
    setup_render_settings(render_high)
    
    # Select vertices and joints based on render target
    verts_list = obj_verts_list_filtered if render_target == TARGET_FLAG_REFINE else obj_verts_list_original
    
    num_frames = len(verts_list)*2-1
    setup_animation_settings(num_frames)
    
    # Create joints and bones if needed
    if render_target != TARGET_FLAG_NONE:
        p1_joints = p1_jnts_refine if render_target == TARGET_FLAG_REFINE else p1_jnts_input
        p2_joints = p2_jnts_refine if render_target == TARGET_FLAG_REFINE else p2_jnts_input
        p1_spheres, p2_spheres, p1_bones, p2_bones = create_joints_and_bones(p1_joints, p2_joints)
    
    # Create object meshes
    create_object_meshes(verts_list, obj_faces_list)
    
    # Update joint and bone positions
    if render_target != TARGET_FLAG_NONE:
        print("Updating joint positions and bones...")
        for frame_num in range(1, len(verts_list)):
            update_joints_and_bones(frame_num, p1_joints, p2_joints, p1_spheres, p2_spheres, p1_bones, p2_bones)
    
    # Create output directory and render
    os.makedirs(VIDEO_DIR, exist_ok=True)
    video_dir = os.path.join(VIDEO_DIR, 'prim_' + os.path.splitext(os.path.basename(obj_folder))[0])
    
    camera_settings = prepare_camera_settings(p1_jnts_input[:,0,:], p2_jnts_input[:,0,:], camera_no)
    render_animation(video_dir, render_target, camera_settings, num_frames)

if __name__ == "__main__":
    main()