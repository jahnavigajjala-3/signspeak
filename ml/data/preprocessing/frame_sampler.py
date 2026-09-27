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

def sample_video_frames(
    video_path: str,
    num_samples: int = 16,
    image_size: int = 224
) -> torch.Tensor:

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("The video cannot be opened")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    indices = sample_frame_indices(total_frames, num_samples)

    frames = []

    for index in indices:
        frame = read_video_frame(video_path, int(index))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_size, image_size))

        frame = frame.astype(np.float32) / 255.0

        frame = torch.from_numpy(frame)
        frame = frame.permute(2, 0, 1)

        frames.append(frame)

    video_tensor = torch.stack(frames)

    return video_tensor