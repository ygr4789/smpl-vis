import bpy
import os
import sys
import threading

from contextlib import contextmanager
from visualize.const import *

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
    bpy.context.scene.cycles.samples = 1024
    bpy.context.scene.cycles.use_denoising = False # device issue
    bpy.context.scene.cycles.use_adaptive_sampling = True
    bpy.context.scene.cycles.adaptive_threshold = 0.1
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.use_nodes = False
    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.render.use_sequencer = True
    bpy.context.scene.render.film_transparent = False

def setup_animation_settings(num_frames):
    """Configure animation and frame settings"""
    bpy.context.scene.render.fps = 30
    bpy.context.scene.render.fps_base = 1
    bpy.context.scene.frame_start = 1
    # Double frames for interpolation
    bpy.context.scene.frame_end = num_frames
    
def render_animation(video_dir, render_target, camera_settings, num_frames):
    """Render animation from different camera angles"""
    for camera_setting in camera_settings:
        video_name = video_name_per_flag[render_target] + "_" + camera_setting['text'] + ".mp4"
        os.makedirs(video_dir, exist_ok=True)
        bpy.context.scene.render.filepath = os.path.join(video_dir, video_name)
        bpy.context.scene.camera.location = camera_setting['location']
        bpy.context.scene.camera.rotation_euler = camera_setting['rotation']
        
        print(f"Rendering {num_frames} frames for {camera_setting['text']}...")
        with stdout_redirected(keyword="Fra:", on_match=lambda line: line[:-1].encode()):
            bpy.ops.render.render(animation=True)
        print()
        print(f"Saved to {video_dir} for {video_name_per_flag[render_target]} {camera_setting['text']}")