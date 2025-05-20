import bpy
import os
import sys
import threading
import argparse
from contextlib import contextmanager
from visualize.const import *
import numpy as np
import math
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

def parse_arguments():
    # Get all arguments after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    # Create argument parser
    parser = argparse.ArgumentParser(description='Render SMPL visualization in Blender')
    parser.add_argument('-i', '--input', type=str, help='Path to obj output folder (input)')
    parser.add_argument('-o', '--output', type=str, help='Path to output video folder (output)')
    parser.add_argument('-q', '--high', action='store_true', help='Use high quality rendering settings')
    parser.add_argument('-t', '--target', type=int, choices=list(keys_to_render_per_flag.keys()),
                        help='Render target: 0=object only, 1=input motion, 2=refined motion(default), ... (see const.py)', default=TARGET_FLAG_REFINE)
    parser.add_argument('-c', '--camera', type=int, help='Camera number, default=-1 for all cameras', default=-1)
    parser.add_argument('-sc', '--scene', type=int, help='Scene number, default=0 for no furnitures', default=0)
    parser.add_argument('-s', '--soft', action='store_true', help='Use soft material')
    
    return parser.parse_args(argv)

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

def cleanup_existing_objects():
    """Hide existing mesh objects except Plane"""
    sample_collection = bpy.data.collections.get('Sample')
    if sample_collection:
        for obj in sample_collection.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

def setup_keyframes(obj, frame_num):
    """Set up keyframes for visibility of object"""
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

def setup_background_scene(scene_no):
    """Setup background scene"""
    scenes_collection = bpy.data.collections.get('Scenes')
    if not scenes_collection:
        print("Warning: 'Scenes' collection not found")
        return
    if f'Scene{scene_no}' not in [c.name for c in scenes_collection.children]:
        print(f"Warning: 'Scene{scene_no}' not found in 'Scenes' collection")
        return
    
    background_objects = []
    for scene_collection in scenes_collection.children:
        scene_collection.hide_render = scene_collection.name != f'Scene{scene_no}'
        background_objects.extend([obj for obj in scene_collection.objects if obj.type == 'MESH'])
    
    floor_obj = bpy.data.objects.get('Floor')
    if floor_obj:
        background_objects.append(floor_obj)

    for obj in background_objects:
        bpy.context.scene.cursor.location = (0, 0, 0)
        with bpy.context.temp_override(selected_editable_objects=[obj]):
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    
    return background_objects

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
        # bpy.context.scene.render.resolution_x = 1920
        # bpy.context.scene.render.resolution_y = 1080
        # bpy.context.scene.render.resolution_percentage = 100
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
    # bpy.context.scene.cycles.samples = 1024
    bpy.context.scene.cycles.samples = 256
    bpy.context.scene.cycles.use_denoising = False # device issue
    bpy.context.scene.cycles.use_adaptive_sampling = True
    bpy.context.scene.cycles.adaptive_threshold = 0.1
    # bpy.context.scene.render.resolution_x = 1920
    # bpy.context.scene.render.resolution_y = 1080
    # bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.resolution_x = 1280
    bpy.context.scene.render.resolution_y = 720
    bpy.context.scene.render.resolution_percentage = 50
    bpy.context.scene.use_nodes = False
    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.render.use_sequencer = True
    bpy.context.scene.render.film_transparent = False

def setup_animation_settings(num_frames):
    """Configure animation and frame settings"""
    bpy.context.scene.render.fps = 30
    bpy.context.scene.render.fps_base = 1
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = num_frames

def setup_camera_setting(camera_setting):
    bpy.context.scene.camera.location = camera_setting['cam_location']
    bpy.context.scene.camera.rotation_euler = camera_setting['cam_rotation']
    
    center = camera_setting['center']
    angle = camera_setting['angle']
    
    scenes_collection = bpy.data.collections.get('Scenes')
    background_objects = []
    for scene_collection in scenes_collection.children:
        background_objects.extend([obj for obj in scene_collection.objects if obj.type == 'MESH'])
    
    floor_obj = bpy.data.objects.get('Floor')
    if floor_obj:
        background_objects.append(floor_obj)

    for obj in background_objects:
        obj.location = center
        obj.rotation_euler = (0, 0, angle)
        
    sun = bpy.data.objects.get('Sun')
    light_rotation = (math.radians(30), 0, angle + math.radians(20))
    if sun:
        sun.rotation_euler = light_rotation

def render_animation(video_dir, render_target, camera_settings, num_frames):
    """Render animation from different camera angles"""
    for camera_setting in camera_settings:
        video_name = video_name_per_flag[render_target] + "_" + camera_setting['text'] + ".mp4"
        os.makedirs(video_dir, exist_ok=True)
        bpy.context.scene.render.filepath = os.path.join(video_dir, video_name)
        setup_camera_setting(camera_setting)
        
        print(f"Rendering {num_frames} frames for {camera_setting['text']}...")
        with stdout_redirected(keyword="Fra:", on_match=lambda line: line[:-1].encode()):
            bpy.ops.render.render(animation=True)
        print()
        print(f"Saved to {video_dir} for {video_name_per_flag[render_target]} {camera_setting['text']}")