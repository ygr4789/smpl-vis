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

def parse_arguments():
    # Get all arguments after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    # Create argument parser
    parser = argparse.ArgumentParser(description='Render SMPL visualization in Blender')
    parser.add_argument('data_file', type=str, help='Path to pkl data file')
    parser.add_argument('-q', '--high', action='store_true', help='Use high quality rendering settings')
    parser.add_argument('-t', '--target', type=int, choices=[0, 1, 2], 
                       help='Render target: 0=object only, 1=input motion, 2=refined motion', 
                       default=TARGET_FLAG_REFINE)
    parser.add_argument('-c', '--camera', type=int, help='Camera number, -1 for all cameras', default=-1)
    
    return parser.parse_args(argv)

@contextmanager
def stdout_redirected(keyword=None, on_match=None):
    """
    Redirect stdout to a pipe and scan it live for a keyword.
    If found, call `on_match(line)` or print it.
    """
    original_fd = sys.stdout.fileno()
    saved_fd = os.dup(original_fd)

    read_fd, write_fd = os.pipe()

    def reader():
        with os.fdopen(read_fd) as read_pipe:
            string = ""
            for line in iter(read_pipe.readline, ''):
                if keyword is not None and keyword in line and on_match:
                    os.write(saved_fd, b'\r')
                    os.write(saved_fd, b' ' * len(string))
                    os.write(saved_fd, b'\r')
                    string = on_match(line)
                    os.write(saved_fd, string)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    os.dup2(write_fd, original_fd)
    try:
        yield
    finally:
        os.dup2(saved_fd, original_fd)
        os.close(write_fd)
        thread.join()
        os.close(saved_fd)

def setup_render_settings(render_high):
    """Configure render settings based on quality mode"""
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
    bpy.context.scene.render.ffmpeg.format = 'MPEG4'
    bpy.context.scene.render.ffmpeg.codec = 'H264'

    if render_high:
        setup_high_quality_settings()
    else:
        setup_low_quality_settings()

def setup_low_quality_settings():
    """Configure settings for fast, low-quality rendering"""
    
    if bpy.app.version >= (4, 2, 0):
        bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
        bpy.context.scene.render.resolution_x = 3840
        bpy.context.scene.render.resolution_y = 2160
        bpy.context.scene.render.resolution_percentage = 100
    else:
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        bpy.context.scene.render.resolution_x = 1280
        bpy.context.scene.render.resolution_y = 720
        bpy.context.scene.render.resolution_percentage = 50

    # Use hasattr to avoid attribute errors
    eevee = getattr(bpy.context.scene, 'eevee', None)
    if eevee:
        if hasattr(eevee, 'taa_render_samples'):
            eevee.taa_render_samples = 16
        for attr in ['use_soft_shadows', 'use_bloom', 'use_ssr', 'use_ssr_refraction']:
            if hasattr(eevee, attr):
                setattr(eevee, attr, False)

    bpy.context.scene.use_nodes = False
    bpy.context.scene.render.use_compositing = False
    bpy.context.scene.render.use_sequencer = False
    bpy.context.scene.render.film_transparent = False

def setup_high_quality_settings():
    """Configure settings for high-quality rendering"""
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 256
    bpy.context.scene.cycles.use_denoising = False # device issue
    bpy.context.scene.cycles.use_adaptive_sampling = True
    bpy.context.scene.cycles.adaptive_threshold = 0.1
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.use_nodes = True
    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.render.use_sequencer = True

def setup_animation_settings(num_frames):
    """Configure animation and frame settings"""
    bpy.context.scene.render.fps = 30
    bpy.context.scene.render.fps_base = 1
    bpy.context.scene.frame_start = 1
    # Double frames for interpolation
    bpy.context.scene.frame_end = num_frames * 2 - 1

def cleanup_existing_objects():
    """Remove existing mesh objects except Plane"""
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.name != 'Plane':
            bpy.data.objects.remove(obj, do_unlink=True)

def create_sphere_for_joint(material, joint_idx, radius=0.05):
    """Create sphere object for joint visualization"""
    # Set different radius for each joint
    joint_radius = joint_radii.get(joint_idx, radius)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=joint_radius)
    sphere = bpy.context.active_object
    sphere.data.materials.append(bpy.data.materials[material])
    return sphere

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

def load_info(data_file):
    """Load root locations from numpy file if available"""
    root_locs_path = os.path.join(data_file, INFO_FILE_NAME)
    if os.path.exists(root_locs_path):
        info = np.load(root_locs_path, allow_pickle=True).item()
        data_type = info[INFO_TYPE]
        root_loc1 = info[INFO_ROOT_LOC_P1]
        root_loc2 = info[INFO_ROOT_LOC_P2]
        cam_T = info[INFO_CAM]
        return data_type, root_loc1, root_loc2, cam_T
    else:
        raise FileNotFoundError(f"No info file found at: {root_locs_path}")

def create_bone_cone(joint1_pos, joint2_pos, material, bone_idx, radius=0.03):
    """Create a cone object connecting two joints to represent a bone"""
    # Calculate direction and length
    direction = joint2_pos - joint1_pos
    length = np.linalg.norm(direction)
    direction = direction / length
    
    # Create cone
    bpy.ops.mesh.primitive_cone_add(radius1=radius, radius2=0, depth=length)
    cone = bpy.context.active_object
    
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

def load_data(data_file):
    """Load and validate data from npz file"""
    npz_file = os.path.splitext(data_file)[0] + '.npz'
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

def render_animation(data_path, render_target, camera_settings, num_frames):
    """Render animation from different camera angles"""
    for camera_setting in camera_settings:
        video_dir = os.path.join(data_path, VIDEO_NAMES[render_target])
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
    data_file = args.data_file
    render_high = args.high
    render_target = args.target
    camera_no = args.camera
    
    # Load data
    data = load_data(data_file)
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
    setup_animation_settings(len(verts_list))
    
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
    data_path = os.path.join(VIDEO_DIR, 'prim_' + os.path.splitext(os.path.basename(data_file))[0])
    
    camera_settings = prepare_camera_settings(p1_jnts_input[:,0,:], p2_jnts_input[:,0,:], camera_no)
    render_animation(data_path, render_target, camera_settings, len(verts_list)*2-1)

if __name__ == "__main__":
    main()