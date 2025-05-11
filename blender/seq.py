import bpy
import os

# Set the path to your OBJ files
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('obj_folder', type=str, help='Path to obj output folder')
parser.add_argument('-l', '--low', action='store_true', help='Use low quality fast rendering settings')
args = parser.parse_args()

obj_folder = args.obj_folder
blender_path = "blender/scene.blend"

os.makedirs("video", exist_ok=True)
video_path = os.path.join("video", os.path.basename(obj_folder) + ".mp4")

# Load scene.blend file
bpy.ops.wm.open_mainfile(filepath=blender_path)

# Set up rendering settings
bpy.context.scene.render.film_transparent = True
bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
bpy.context.scene.render.ffmpeg.format = 'MPEG4'
bpy.context.scene.render.ffmpeg.codec = 'H264'
bpy.context.scene.render.filepath = video_path

if args.low:
    # Use Eevee for fast rendering
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    # Reduce quality settings
    bpy.context.scene.eevee.taa_render_samples = 16
    bpy.context.scene.eevee.use_soft_shadows = False
    bpy.context.scene.eevee.use_bloom = False
    bpy.context.scene.eevee.use_ssr = False
    bpy.context.scene.eevee.use_ssr_refraction = False
    # Reduce output resolution
    bpy.context.scene.render.resolution_x = 1280
    bpy.context.scene.render.resolution_y = 720
    bpy.context.scene.render.resolution_percentage = 50
    # Use view layer directly
    bpy.context.scene.use_nodes = False
    # Disable compositing
    bpy.context.scene.render.use_compositing = False
    bpy.context.scene.render.use_sequencer = False
else:
    # Use Cycles with normal settings
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 256
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.use_adaptive_sampling = True
    bpy.context.scene.cycles.adaptive_threshold = 0.1
    # Full resolution
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.resolution_percentage = 100
    # Enable compositing for high quality
    bpy.context.scene.use_nodes = True
    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.render.use_sequencer = True

# Set frame rate to 30fps
bpy.context.scene.render.fps = 30
bpy.context.scene.render.fps_base = 1

# Set frame range
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = len(os.listdir(os.path.join(obj_folder, "obj")))  # Adjust based on your number of frames

# Remove any existing mesh objects except Plane
for obj in bpy.data.objects:
    if obj.type == 'MESH' and obj.name != 'Plane':
        bpy.data.objects.remove(obj, do_unlink=True)

# Create list of obj paths and names
objs = ["obj", "p1", "p2"]
materials = ["Yellow", "Red", "Blue"]  # Materials in corresponding order
obj_paths = [os.path.join(obj_folder, obj) for obj in objs]
obj_files = [sorted(f for f in os.listdir(path) if f.endswith('.obj')) for path in obj_paths]

# Import each obj at a specific frame
for i, files in enumerate(zip(*obj_files)):
    frame_num = i + 1
    
    # Import all objects for this frame
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
    
    # Set keyframes for all objects
    for obj in imported_objs:
        obj.hide_render = True
        obj.hide_viewport = True
        obj.keyframe_insert(data_path="hide_render", frame=1)
        obj.keyframe_insert(data_path="hide_viewport", frame=1)
        
        obj.hide_render = False
        obj.hide_viewport = False
        obj.keyframe_insert(data_path="hide_render", frame=frame_num)
        obj.keyframe_insert(data_path="hide_viewport", frame=frame_num)
        
        obj.hide_render = True
        obj.hide_viewport = True
        obj.keyframe_insert(data_path="hide_render", frame=frame_num + 1)
        obj.keyframe_insert(data_path="hide_viewport", frame=frame_num + 1)

# After all the keyframes are set, render the animation
bpy.ops.render.render(animation=True)