import bpy
import os
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from blender.camera import prepare_camera_settings
from blender.utils import setup_render_settings, setup_animation_settings, render_animation, cleanup_existing_objects, parse_arguments, setup_keyframes, setup_background_scene
from visualize.const import *
from blender.prim import *

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


def load_info(obj_folder):
    """Load root locations from numpy file if available"""
    root_locs_path = os.path.join(obj_folder, INFO_FILE_NAME)
    if os.path.exists(root_locs_path):
        info = np.load(root_locs_path, allow_pickle=True).item()
        root_loc1 = info[INFO_ROOT_LOC_P1]
        root_loc2 = info[INFO_ROOT_LOC_P2]
        return root_loc1, root_loc2
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

def create_joints_and_bones(p1_joints, p2_joints, material1, material2):
    """Create joint spheres and bone cones for both characters"""
    # Create spheres for p1 joints
    p1_spheres = [create_sphere_for_joint(material1, i) for i in range(p1_joints.shape[1])]
    # Create spheres for p2 joints  
    p2_spheres = [create_sphere_for_joint(material2, i) for i in range(p2_joints.shape[1])]
    
    # Create bones for p1
    p1_bones = []
    for bone_idx, (joint1, joint2) in bone_pair.items():
        cone = create_bone_cone(p1_joints[0, joint1], p1_joints[0, joint2], material1, bone_idx, bone_radius[bone_idx])
        p1_bones.append((cone, joint1, joint2))
        
    # Create bones for p2
    p2_bones = []
    for bone_idx, (joint1, joint2) in bone_pair.items():
        cone = create_bone_cone(p2_joints[0, joint1], p2_joints[0, joint2], material2, bone_idx, bone_radius[bone_idx])
        p2_bones.append((cone, joint1, joint2))
        
    return p1_spheres, p2_spheres, p1_bones, p2_bones

def create_object_meshes(verts_list, obj_faces_list, material):
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
        obj = create_mesh_for_frame(verts, obj_faces_list, frame_num, material)
        setup_keyframes(obj, frame_num)

def update_joints_and_bones(frame_num, p1_joints, p2_joints, p1_spheres, p2_spheres, p1_bones, p2_bones):
    """Update joint and bone positions for a frame"""
    anim_frame = frame_num * 2 + 1
    
    # Update p1 joints and bones
    for sphere, pos in zip(p1_spheres, p1_joints[frame_num]):
        sphere.location = pos
        sphere.keyframe_insert(data_path="location", frame=anim_frame)
    
    for cone, joint1, joint2 in p1_bones:
        joint1_pos = p1_joints[frame_num, joint1]
        joint2_pos = p1_joints[frame_num, joint2]
        update_bone_position(cone, joint1_pos, joint2_pos, anim_frame)
        
    # Update p2 joints and bones
    for sphere, pos in zip(p2_spheres, p2_joints[frame_num]):
        sphere.location = pos
        sphere.keyframe_insert(data_path="location", frame=anim_frame)
    
    for cone, joint1, joint2 in p2_bones:
        joint1_pos = p2_joints[frame_num, joint1]
        joint2_pos = p2_joints[frame_num, joint2]
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

def load_data_for_target(obj_folder, render_target):
    """Load data based on render target flag"""
    data = load_data(obj_folder)
    keys_to_load = keys_to_render_per_flag[render_target]
    keys_to_load.append(KEY_OBJ_FACES)
    
    # Initialize data dictionary
    loaded_data = {}
    
    # Load data for each key
    for key in keys_to_load:
        if key in data:
            loaded_data[key] = data[key]
        else:
            raise KeyError(f"Required key '{key}' not found in data file")
            
    return loaded_data

def prepare_render_data(data, render_target):
    vert_keys = [key for key in data.keys() if 'verts_list' in key.lower()]
    if len(vert_keys) != 1:
        raise ValueError(f"Expected exactly one vert list key, found {len(vert_keys)}: {vert_keys}")
    verts_list = data[vert_keys[0]]
    num_frames = len(verts_list)
    
    obj_faces_list = data[KEY_OBJ_FACES]
    
    p1_joints = None
    p2_joints = None
    
    if render_target != TARGET_FLAG_NONE:
        p1_keys = [key for key in data.keys() if 'p1' in key.lower()]
        p2_keys = [key for key in data.keys() if 'p2' in key.lower()]
        
        if len(p1_keys) != 1:
            raise ValueError(f"Expected exactly one P1 joint key, found {len(p1_keys)}: {p1_keys}")
        if len(p2_keys) != 1:
            raise ValueError(f"Expected exactly one P2 joint key, found {len(p2_keys)}: {p2_keys}")
            
        p1_joints = data[p1_keys[0]]
        p2_joints = data[p2_keys[0]]
        num_frames = min(num_frames,len(p1_joints), len(p2_joints))
    
    return verts_list, obj_faces_list, p1_joints, p2_joints, num_frames

def main():
    args = parse_arguments()
    obj_folder = args.input
    video_dir = args.output
    render_high = args.high
    render_target = args.target
    camera_no = args.camera
    soft = args.soft
    scene_no = args.scene
    
    root_loc1, root_loc2 = load_info(obj_folder)
    
    # Load data based on render target
    print(f"Loading data for target flag {render_target}...")
    data = load_data_for_target(obj_folder, render_target)
    
    # Load scene and setup
    bpy.ops.wm.open_mainfile(filepath=BLENDER_PATH)
    cleanup_existing_objects()
    setup_render_settings(render_high)
    setup_background_scene(scene_no)
    
    # Prepare render data
    verts_list, obj_faces_list, p1_joints, p2_joints, num_frames = prepare_render_data(data, render_target)
    materials = ["Yellow", "Red", "Blue"] if not soft else ["Yellow_soft", "Red_soft", "Blue_soft"]
        
    setup_animation_settings(num_frames*2-1)
    
    # Create joints and bones if needed
    if render_target != TARGET_FLAG_NONE:
        # Create joints and bones
        p1_spheres, p2_spheres, p1_bones, p2_bones = create_joints_and_bones(p1_joints, p2_joints, materials[1], materials[2])
    
    # Create object meshes
    create_object_meshes(verts_list, obj_faces_list, materials[0])
    
    # Update joint and bone positions if they exist
    if p1_joints is not None and p2_joints is not None:
        print("Updating joint positions and bones...")
        for frame_num in range(num_frames):
            update_joints_and_bones(frame_num, p1_joints, p2_joints, p1_spheres, p2_spheres, p1_bones, p2_bones)
    
    # Create output directory and render
    if video_dir is None:
        data_name = os.path.basename(obj_folder)
        video_dir = os.path.join(VIDEO_DIR, 'prim_' + data_name)
    os.makedirs(video_dir, exist_ok=True)
    
    # Render animation
    camera_settings = prepare_camera_settings(root_loc1, root_loc2, camera_no)
    render_animation(video_dir, render_target, camera_settings, num_frames*2-1)

if __name__ == "__main__":
    main()