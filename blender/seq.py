import bpy
import os
import argparse
import sys
import numpy as np
import threading

from contextlib import contextmanager
from blender.camera import prepare_camera_settings
from blender.const import *

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

def setup_animation_settings(obj_folder):
    """Configure animation and frame settings"""
    bpy.context.scene.render.fps = 30
    bpy.context.scene.render.fps_base = 1
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = len(os.listdir(os.path.join(obj_folder, OBJ_OBJ_ORIGINAL)))

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
    setup_animation_settings(obj_folder)
    
    # Process each frame
    for i, files in enumerate(zip(*obj_files)):
        frame_num = i + 1
        import_and_setup_frame(obj_paths, files, materials, frame_num)
        progress = frame_num / len(obj_files[0]) * 100
        print(f"\rImporting objs: [{('=' * int(progress/2)).ljust(50)}] {progress:.1f}%", end='', flush=True)
    print()
    
    # Create output directory
    os.makedirs(VIDEO_DIR, exist_ok=True)
    data_path = os.path.join(VIDEO_DIR, os.path.basename(obj_folder))
    
    # Render animation
    camera_settings = prepare_camera_settings(root_loc1, root_loc2, camera_no, cam_T)
    for camera_setting in camera_settings:
        video_dir = os.path.join(data_path, VIDEO_NAMES[render_target])
        video_name = VIDEO_NAMES[render_target] + "_" + camera_setting['text'] + ".mp4"
        os.makedirs(video_dir, exist_ok=True)
        bpy.context.scene.render.filepath = os.path.join(video_dir, video_name)
        bpy.context.scene.camera.location = camera_setting['location']
        bpy.context.scene.camera.rotation_euler = camera_setting['rotation']
        
        print(f"Rendering {len(obj_files[0])} frames for {camera_setting['text']}...")
        with stdout_redirected(keyword="Fra:", on_match=lambda line: line[:-1].encode()):
            bpy.ops.render.render(animation=True)
        print()
        print(f"Saved to {video_dir} for {camera_setting['text']}")

if __name__ == "__main__":
    main()