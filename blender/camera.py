import bpy
import mathutils
import numpy as np
import math

from visualize.const import *

def get_camera_params(camera_no):
    camera_params = [
        # azimuth, text
        [0,      "cam00"],
        [20,     "cam01"],
        [-20,    "cam02"],
        [180,    "cam03"],
        [200,    "cam04"],
        [160,    "cam05"],
    ]
    if camera_no == -1: # all cameras
        return camera_params
    elif camera_no < len(camera_params):
        return [camera_params[camera_no]]
    else:
        raise ValueError(f"Camera no. {camera_no} does not exist")

def prepare_camera_settings(root_loc1, root_loc2, camera_no):
    """Adjust camera position by doubling its location vector"""
    root_loc1_mean = np.mean(root_loc1, axis=0)
    root_loc2_mean = np.mean(root_loc2, axis=0)
    center = (root_loc1_mean + root_loc2_mean) / 2
    AB = root_loc2_mean - root_loc1_mean
    center[2] = 0
    AB[2] = 0
    
    camera_params = get_camera_params(camera_no)
    camera_settings = []
    
    initial_camera = bpy.context.scene.camera
    initial_location = initial_camera.location
    initial_rotation = initial_camera.rotation_euler
    
    initial_height = initial_location[2]
    initial_distance = math.sqrt(initial_location[0]**2 + initial_location[1]**2)
    initial_elevation = math.pi / 2 - initial_rotation[0]
    
    initial_dir = initial_location
    initial_dir[2] = 0
    initial_dir = initial_dir.normalized()
    
    for camera_param in camera_params:
      azimuth, text = camera_param
      up = mathutils.Vector((0, 0, 1))
      cam_dir = up.cross(AB).normalized()
      
      azimuth = math.radians(azimuth)
      rot_azimuth = mathutils.Matrix.Rotation(azimuth, 3, 'Z')
      cam_dir = rot_azimuth @ cam_dir
      angle = math.atan2(initial_dir.cross(cam_dir).dot(up), initial_dir.dot(cam_dir))
      
      cam_location = center + cam_dir * initial_distance + up * initial_height
      
      right = cam_dir.cross(up).normalized()
      rot_elevation = mathutils.Matrix.Rotation(initial_elevation, 3, right)
      cam_dir = rot_elevation @ cam_dir
      
      cam_rotation = (-cam_dir).to_track_quat('-Z', 'Y').to_euler()
      
      camera_setting = {
          'cam_location': cam_location,
          'cam_rotation': cam_rotation,
          'center': center,
          'angle': angle,
          'text': text
      }
      camera_settings.append(camera_setting)
    
    
    return camera_settings
