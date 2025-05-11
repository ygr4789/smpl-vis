import bpy
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('obj_folder', type=str, help='Path to obj output folder')
    parser.add_argument('-l', '--low', action='store_true', help='Use low quality fast rendering settings')
    return parser.parse_args()

def setup_render_settings(args, video_path):
    """Configure render settings based on quality mode"""
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
    bpy.context.scene.render.ffmpeg.format = 'MPEG4'
    bpy.context.scene.render.ffmpeg.codec = 'H264'
    bpy.context.scene.render.filepath = video_path

    if args.low:
        setup_low_quality_settings()
    else:
        setup_high_quality_settings()

def setup_low_quality_settings():
    """Configure settings for fast, low-quality rendering"""
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    bpy.context.scene.eevee.taa_render_samples = 16
    bpy.context.scene.eevee.use_soft_shadows = False
    bpy.context.scene.eevee.use_bloom = False
    bpy.context.scene.eevee.use_ssr = False
    bpy.context.scene.eevee.use_ssr_refraction = False
    bpy.context.scene.render.resolution_x = 1280
    bpy.context.scene.render.resolution_y = 720
    bpy.context.scene.render.resolution_percentage = 50
    bpy.context.scene.use_nodes = False
    bpy.context.scene.render.use_compositing = False
    bpy.context.scene.render.use_sequencer = False

def setup_high_quality_settings():
    """Configure settings for high-quality rendering"""
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 256
    bpy.context.scene.cycles.use_denoising = True
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
    bpy.context.scene.frame_end = len(os.listdir(os.path.join(obj_folder, "obj")))

def cleanup_existing_objects():
    """Remove existing mesh objects except Plane"""
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.name != 'Plane':
            bpy.data.objects.remove(obj, do_unlink=True)

def import_and_setup_frame(obj_paths, files, materials, frame_num):
    """Import objects for a specific frame and set up their keyframes"""
    imported_objs = []
    for obj_path, file_name, material in zip(obj_paths, files, materials):
        file_path = os.path.join(obj_path, file_name)
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

def main():
    args = parse_arguments()
    obj_folder = args.obj_folder
    blender_path = "blender/scene.blend"
    
    # Create output directory
    os.makedirs("video", exist_ok=True)
    video_path = os.path.join("video", os.path.basename(obj_folder) + ".mp4")
    
    # Load scene and setup
    bpy.ops.wm.open_mainfile(filepath=blender_path)
    setup_render_settings(args, video_path)
    setup_animation_settings(obj_folder)
    cleanup_existing_objects()
    
    # Prepare object paths and materials
    objs = ["obj", "p1", "p2"]
    materials = ["Yellow", "Red", "Blue"]
    obj_paths = [os.path.join(obj_folder, obj) for obj in objs]
    obj_files = [sorted(f for f in os.listdir(path) if f.endswith('.obj')) for path in obj_paths]
    
    # Process each frame
    for i, files in enumerate(zip(*obj_files)):
        frame_num = i + 1
        import_and_setup_frame(obj_paths, files, materials, frame_num)
    
    # Render animation
    bpy.ops.render.render(animation=True)

if __name__ == "__main__":
    main()