import bpy
import os
import argparse
import contextlib
import numpy as np
import math
import threading
import mathutils

import sys
from contextlib import contextmanager

def parse_arguments():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]  
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument('obj_folder', type=str, help='Path to obj output folder')
    parser.add_argument('-l', '--low', action='store_true', help='Use low quality fast rendering settings')
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

def setup_render_settings(args, video_path):
    """Configure render settings based on quality mode"""
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
    bpy.context.scene.render.ffmpeg.format = 'MPEG4'
    bpy.context.scene.render.ffmpeg.codec = 'H264'
    bpy.context.scene.render.filepath = video_path

    # if args.low:
    #     setup_low_quality_settings()
    # else:
    #     setup_high_quality_settings()
    setup_settings()

def setup_settings():
    """Safe settings for rendering under EEVEE"""
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'  # Or EEVEE_NEXT, with caution

    # Use hasattr to avoid attribute errors
    eevee = getattr(bpy.context.scene, 'eevee', None)
    if eevee:
        if hasattr(eevee, 'taa_render_samples'):
            eevee.taa_render_samples = 16
        for attr in ['use_soft_shadows', 'use_bloom', 'use_ssr', 'use_ssr_refraction']:
            if hasattr(eevee, attr):
                setattr(eevee, attr, False)

    bpy.context.scene.render.resolution_x = 3840
    bpy.context.scene.render.resolution_y = 2160
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.use_nodes = False
    bpy.context.scene.render.use_compositing = False
    bpy.context.scene.render.use_sequencer = False

# def setup_low_quality_settings():
#     """Configure settings for fast, low-quality rendering"""
#     bpy.context.scene.render.engine = 'BLENDER_EEVEE'
#     bpy.context.scene.eevee.taa_render_samples = 16
#     bpy.context.scene.eevee.use_soft_shadows = False
#     bpy.context.scene.eevee.use_bloom = False
#     bpy.context.scene.eevee.use_ssr = False
#     bpy.context.scene.eevee.use_ssr_refraction = False
#     bpy.context.scene.render.resolution_x = 1280
#     bpy.context.scene.render.resolution_y = 720
#     bpy.context.scene.render.resolution_percentage = 50
#     bpy.context.scene.use_nodes = False
#     bpy.context.scene.render.use_compositing = False
#     bpy.context.scene.render.use_sequencer = False
#     # bpy.context.scene.render.engine = 'CYCLES'
#     # bpy.context.scene.cycles.samples = 256
#     # bpy.context.scene.cycles.use_denoising = True
#     # bpy.context.scene.cycles.use_adaptive_sampling = True
#     # bpy.context.scene.cycles.adaptive_threshold = 0.1
#     # bpy.context.scene.render.resolution_x = 1280
#     # bpy.context.scene.render.resolution_y = 720
#     # bpy.context.scene.render.resolution_percentage = 50
#     # bpy.context.scene.use_nodes = True
#     # bpy.context.scene.render.use_compositing = True
#     # bpy.context.scene.render.use_sequencer = True

# def setup_high_quality_settings():
#     """Configure settings for high-quality rendering"""
#     bpy.context.scene.render.engine = 'CYCLES'
#     bpy.context.scene.cycles.samples = 256
#     bpy.context.scene.cycles.use_denoising = True
#     bpy.context.scene.cycles.use_adaptive_sampling = True
#     bpy.context.scene.cycles.adaptive_threshold = 0.1
#     bpy.context.scene.render.resolution_x = 1920
#     bpy.context.scene.render.resolution_y = 1080
#     bpy.context.scene.render.resolution_percentage = 100
#     bpy.context.scene.use_nodes = True
#     bpy.context.scene.render.use_compositing = True
#     bpy.context.scene.render.use_sequencer = True

def setup_animation_settings(obj_folder):
    """Configure animation and frame settings"""
    bpy.context.scene.render.fps = 30
    bpy.context.scene.render.fps_base = 1
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = len(os.listdir(os.path.join(obj_folder, "obj")))

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

def load_root_locations(obj_folder):
    """Load root locations from numpy file if available"""
    root_locs_path = os.path.join(obj_folder, "root_locs.npy")
    if os.path.exists(root_locs_path):
        root_locs = np.load(root_locs_path, allow_pickle=True).item()
        return root_locs
    else:
        print("No root locations file found at:", root_locs_path)
        return None

def adjust_camera_position(root_locs, azimuth=5, elevation=30, distance=6.0):
    """Adjust camera position by doubling its location vector"""
    root_loc1 = root_locs['root_loc1'].copy()
    root_loc2 = root_locs['root_loc2'].copy()
    root_loc1[:, [1, 2]] = root_loc1[:, [2, 1]]  # y,z -> z,y
    root_loc2[:, [1, 2]] = root_loc2[:, [2, 1]]  # y,z -> z,y
    root_loc1[:, 1] *= -1  # flip y
    root_loc2[:, 1] *= -1  # flip y
    
    cam = bpy.context.scene.camera
    if cam:
        root_loc1_mean = np.mean(root_loc1, axis=0)
        root_loc2_mean = np.mean(root_loc2, axis=0)
        center = (root_loc1_mean + root_loc2_mean) / 2
        
        AB = root_loc2_mean - root_loc1_mean
        up = mathutils.Vector((0, 0, 1))
        cam_dir = mathutils.Vector(AB).cross(up).normalized()
        
        azimuth_rad = math.radians(azimuth)
        elevation_rad = math.radians(elevation)
        rot_azimuth = mathutils.Matrix.Rotation(azimuth_rad, 3, 'Z')
        rot_elevation = mathutils.Matrix.Rotation(-elevation_rad, 3, 'X')
        cam_dir = rot_elevation @ rot_azimuth @ cam_dir
        
        center = mathutils.Vector(center)
        cam.location = center + cam_dir * distance
        
        rot_quat = (-cam_dir).to_track_quat('-Z', 'Y')
        cam.rotation_euler = rot_quat.to_euler()
    else:
        print("No camera found in scene")
        
def main():
    args = parse_arguments()
    obj_folder = args.obj_folder
    blender_path = "blender/scene.blend"
    
    # Create output directory
    os.makedirs("video", exist_ok=True)
    video_path = os.path.join("video", os.path.basename(obj_folder) + ".mp4")
    
    # Load scene and setup
    bpy.ops.wm.open_mainfile(filepath=blender_path)
    cleanup_existing_objects()
    setup_render_settings(args, video_path)
    setup_animation_settings(obj_folder)
    root_locs = load_root_locations(obj_folder)
    if root_locs is not None:
        adjust_camera_position(root_locs)
    
    # Prepare object paths and materials
    objs = ["obj", "p1", "p2"]
    materials = ["Yellow", "Red", "Blue"]
    obj_paths = [os.path.join(obj_folder, obj) for obj in objs]
    obj_files = [sorted(f for f in os.listdir(path) if f.endswith('.obj')) for path in obj_paths]
    
    # Process each frame
    for i, files in enumerate(zip(*obj_files)):
        frame_num = i + 1
        import_and_setup_frame(obj_paths, files, materials, frame_num)
        progress = frame_num / len(obj_files[0]) * 100
        print(f"\rImporting objs: [{('=' * int(progress/2)).ljust(50)}] {progress:.1f}%", end='', flush=True)
    print()
    
    # Render animation
    print(f"Rendering {len(obj_files[0])} frames...")
    with stdout_redirected(keyword="Fra:", on_match=lambda line: line[:-1].encode()):
        bpy.ops.render.render(animation=True)
    print()

if __name__ == "__main__":
    main()