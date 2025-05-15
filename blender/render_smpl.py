import bpy
import os
import argparse
import sys
import numpy as np
import threading

from contextlib import contextmanager
from blender.camera import prepare_camera_settings
from blender.const import *
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
    parser.add_argument('obj_folder', type=str, help='Path to obj output folder')
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

def import_and_setup_frame(obj_paths, files, materials, frame_num):
    """Import objects for a specific frame and set up their keyframes"""
    imported_objs = []
    for i, (obj_path, file_name, material) in enumerate(zip(obj_paths, files, materials)):
        file_path = os.path.join(obj_path, file_name)
        with stdout_redirected():
            bpy.ops.wm.obj_import(filepath=file_path)
        imported_obj = bpy.context.selected_objects[0]
        with bpy.context.temp_override(selected_editable_objects=[imported_obj]):
            bpy.ops.object.shade_smooth()
        imported_obj.data.materials.append(bpy.data.materials[material])
        obj_type = os.path.basename(obj_path)
        imported_obj.name = f"Frame_{frame_num}_{obj_type}"
        imported_objs.append(imported_obj)
        
    setup_keyframes(imported_objs, frame_num)
    return imported_objs

def setup_keyframes(imported_objs, frame_num):
    """Set up keyframes for visibility of imported objects"""
    for obj in imported_objs:
        # Hide at start
        obj.hide_render = True
        obj.hide_viewport = True
        obj.keyframe_insert(data_path="hide_render", frame=1)
        obj.keyframe_insert(data_path="hide_viewport", frame=1)
        
        # Show at current frame
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

def prepare_obj_paths_and_materials(obj_folder, render_target):
    if render_target == TARGET_FLAG_INPUT:
        objs = [OBJ_OBJ_ORIGINAL, OBJ_P1_INPUT, OBJ_P2_INPUT]
        materials = ["Yellow", "Red", "Blue"]
    elif render_target == TARGET_FLAG_REFINE:
        objs = [OBJ_OBJ_FILTERED, OBJ_P1_REFINE, OBJ_P2_REFINE]
        materials = ["Yellow", "Red", "Blue"]
    else:
        objs = [OBJ_OBJ_ORIGINAL]
        materials = ["Yellow"]
    obj_paths = [os.path.join(obj_folder, obj) for obj in objs]
    obj_files = [sorted(f for f in os.listdir(path) if f.endswith('.obj')) for path in obj_paths]
    return obj_paths, obj_files, materials

def main():
    args = parse_arguments()
    obj_folder = args.obj_folder
    render_high = args.high
    render_target = args.target
    camera_no = args.camera
    
    data_type, root_loc1, root_loc2, cam_T = load_info(obj_folder)
    
    if data_type == TYPE_GT and render_target == TARGET_FLAG_NONE:
        print("Skipping GT data with target flag 0")
        return
    
    # Prepare object paths and materials
    obj_paths, obj_files, materials = prepare_obj_paths_and_materials(obj_folder, render_target)
    
    # Load scene and setup
    bpy.ops.wm.open_mainfile(filepath=BLENDER_PATH)
    cleanup_existing_objects()
    setup_render_settings(render_high)
    
    num_frames = len(obj_files[0])
    setup_animation_settings(num_frames)
    
    # Process each frame
    for i, files in enumerate(zip(*obj_files)):
        frame_num = i + 1
        import_and_setup_frame(obj_paths, files, materials, frame_num)
        progress = frame_num / len(obj_files[0]) * 100
        print(f"\rImporting objs: [{('=' * int(progress/2)).ljust(50)}] {progress:.1f}%", end='', flush=True)
    print()
    
    # Create output directory
    os.makedirs(VIDEO_DIR, exist_ok=True)
    data_name = os.path.basename(obj_folder)
    video_dir = os.path.join(VIDEO_DIR, 'smpl_' + data_name)
    
    # Render animation
    camera_settings = prepare_camera_settings(root_loc1, root_loc2, camera_no, cam_T)
    for camera_setting in camera_settings:
        video_name = f"{VIDEO_NAMES[render_target]}_{camera_setting['text']}.mp4"
        os.makedirs(video_dir, exist_ok=True)
        bpy.context.scene.render.filepath = os.path.join(video_dir, video_name)
        bpy.context.scene.camera.location = camera_setting['location']
        bpy.context.scene.camera.rotation_euler = camera_setting['rotation']
        
        print(f"Rendering {num_frames} frames for {camera_setting['text']}...")
        with stdout_redirected(keyword="Fra:", on_match=lambda line: line[:-1].encode()):
            bpy.ops.render.render(animation=True)
        print()
        print(f"Saved to {video_dir} for {camera_setting['text']}")

if __name__ == "__main__":
    main()