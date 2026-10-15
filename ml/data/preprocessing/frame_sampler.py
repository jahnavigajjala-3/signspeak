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

    # If container metadata does not provide frame count (common with browser WebM recordings),
    # count and extract frames dynamically by reading the stream sequentially.
    if total_frames <= 0:
        frames_list = []
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frames_list.append(frame)
        cap.release()

        total_frames = len(frames_list)
        if total_frames <= 0:
            raise ValueError("Video contains no readable frames")

        indices = sample_frame_indices(total_frames, num_samples)
        selected_raw_frames = [frames_list[int(i)] for i in indices]
    else:
        # Try seek-based extraction first, falling back to sequential read if seek fails
        indices = sample_frame_indices(total_frames, num_samples)
        selected_raw_frames = []
        seek_failed = False
        for index in indices:
            try:
                frame = read_video_frame(video_path, int(index))
                if frame is None:
                    seek_failed = True
                    break
                selected_raw_frames.append(frame)
            except Exception:
                seek_failed = True
                break

        if seek_failed:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frames_list = []
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frames_list.append(frame)
            cap.release()

            total_frames = len(frames_list)
            if total_frames <= 0:
                raise ValueError("Video contains no readable frames")

            indices = sample_frame_indices(total_frames, num_samples)
            selected_raw_frames = [frames_list[int(i)] for i in indices]
        else:
            cap.release()

    frames = []
    for frame in selected_raw_frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_size, image_size))
        frame = frame.astype(np.float32) / 255.0
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)
        frames.append(frame_tensor)

    video_tensor = torch.stack(frames)

    return video_tensor