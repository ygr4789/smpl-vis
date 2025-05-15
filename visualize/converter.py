import numpy as np
from trimesh import Trimesh

from abc import ABC, abstractmethod

class converter(ABC):
    @abstractmethod
    def save_obj(self, save_path, frame_idx):
        """Save a single frame's vertices and faces as an obj file.
        
        Args:
            save_path: str, path to save the obj file
            frame_idx: int, index of frame to save
        """
        pass
