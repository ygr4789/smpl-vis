import numpy as np
import math
import mathutils

from visualize.const import *

def get_camera_params(camera_no):
    camera_params = [
        # center_height, azimuth, elevation, distance, text
        [0.7,             10,     30,        7.0,      "cam00"],
        [0.7,             25,     30,        7.0,      "cam01"],
        [0.7,             40,     30,        7.0,      "cam02"],
        [0.7,            -10,     30,        7.0,      "cam03"],
        [0.7,            -25,     30,        7.0,      "cam04"],
        [0.7,            -40,     30,        7.0,      "cam05"],
        [0.7,             10,     10,        6.0,      "cam06"],
        [0.7,             25,     10,        6.0,      "cam07"],
        [0.7,             40,     10,        6.0,      "cam08"],
        [0.7,            -10,     10,        6.0,      "cam09"],
        [0.7,            -25,     10,        6.0,      "cam10"],
        [0.7,            -40,     10,        6.0,      "cam11"],
    ]
    if camera_no == -1: # all cameras
        return camera_params
    elif camera_no < len(camera_params):
        return [camera_params[camera_no]]
    else:
        raise ValueError(f"Camera no. {camera_no} does not exist")

def prepare_camera_settings(root_loc1, root_loc2, camera_no, cam_T=None):
    """Adjust camera position by doubling its location vector"""
    root_loc1_mean = np.mean(root_loc1, axis=0)
    root_loc2_mean = np.mean(root_loc2, axis=0)
    center = (root_loc1_mean + root_loc2_mean) / 2
    
    
    camera_params = get_camera_params(camera_no)
    camera_settings = []
    for camera_param in camera_params:
      center_height, azimuth, elevation, distance, text = camera_param
      center[2] = center_height
      AB = root_loc2_mean - root_loc1_mean
      up = mathutils.Vector((0, 0, 1))
      cam_dir = mathutils.Vector(AB).cross(up).normalized()
      
      azimuth_rad = math.radians(azimuth)
      elevation_rad = math.radians(elevation)
      
      rot_azimuth = mathutils.Matrix.Rotation(azimuth_rad, 3, 'Z')
      cam_dir = rot_azimuth @ cam_dir
      
      right = cam_dir.cross(up).normalized()
      rot_elevation = mathutils.Matrix.Rotation(elevation_rad, 3, right)
      cam_dir = rot_elevation @ cam_dir
      
      cam_location = center + cam_dir * distance
      cam_rotation = (-cam_dir).to_track_quat('-Z', 'Y').to_euler()
      
      camera_setting = {
          'location': cam_location,
          'rotation': cam_rotation,
          'text': text
      }
      camera_settings.append(camera_setting)
    
    
    return camera_settings
