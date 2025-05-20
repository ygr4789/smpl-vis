import bpy
import os
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from blender.camera import prepare_camera_settings
from blender.utils import setup_render_settings, setup_animation_settings, stdout_redirected, render_animation, cleanup_existing_objects, parse_arguments, setup_keyframes, load_info, setup_background_scene
from visualize.const import *

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
        
    for obj in imported_objs:
        setup_keyframes(obj, frame_num)

def prepare_obj_paths_and_materials(obj_folder, render_target, soft):
    objs = [key_path_map[key] for key in keys_to_render_per_flag[render_target]]
    materials = ["Yellow", "Red", "Blue"] if not soft else ["Yellow_soft", "Red_soft", "Blue_soft"]
    materials = materials[:len(objs)]
    
    obj_paths = [os.path.join(obj_folder, obj) for obj in objs]
    obj_files = [sorted(f for f in os.listdir(path) if f.endswith('.obj')) for path in obj_paths]
    
    # Check if all object paths exist
    for path in obj_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Object path does not exist: {path}")
    
    return obj_paths, obj_files, materials

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
    
    # Prepare object paths and materials
    print(f"Preparing for {obj_folder}...")
    obj_paths, obj_files, materials = prepare_obj_paths_and_materials(obj_folder, render_target, soft)
    
    # Load scene and setup
    bpy.ops.wm.open_mainfile(filepath=BLENDER_PATH)
    cleanup_existing_objects()
    setup_render_settings(render_high)
    setup_background_scene(scene_no)
    
    num_frames = min(len(files) for files in obj_files)
    setup_animation_settings(num_frames)
    
    # Process each frame
    for i, files in enumerate(zip(*obj_files)):
        frame_num = i + 1
        import_and_setup_frame(obj_paths, files, materials, frame_num)
        progress = frame_num / len(obj_files[0]) * 100
        print(f"\rImporting objs: [{('=' * int(progress/2)).ljust(50)}] {progress:.1f}%", end='', flush=True)
    print()
    
    # Create output directory
    if video_dir is None:
        data_name = os.path.basename(obj_folder)
        video_dir = os.path.join(VIDEO_DIR, 'smpl_' + data_name)
    os.makedirs(video_dir, exist_ok=True)
    
    # Render animation
    camera_settings = prepare_camera_settings(root_loc1, root_loc2, camera_no)
    render_animation(video_dir, render_target, camera_settings, num_frames)

if __name__ == "__main__":
    main()