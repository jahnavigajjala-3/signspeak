import cv2
import numpy as np
import torch

from ml.data.loaders.video_loader import read_video_frame
def sample_frame_indices(total_frames: int, num_samples: int = 16) -> np.ndarray:
        if total_frames <= 0:
         raise ValueError("total_frames must be positive")

        if num_samples <= 0:
         raise ValueError("num_samples must be positive")
 
        indices = np.linspace(
         0,
         total_frames - 1,
         num=num_samples
         )

        indices = indices.astype(int)

        return indices