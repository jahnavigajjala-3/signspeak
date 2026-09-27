from ml.data.preprocessing.frame_sampler import sample_frame_indices
from ml.data.preprocessing.frame_sampler import sample_video_frames

video_path = "data/raw/how2sign/videos/train/raw_videos/DvKS20WKiX4_22-8-rgb_front.mp4"

indices = sample_frame_indices(300, 16)

print("Indices:", indices)
print("Number of samples:", len(indices))
print("First frame:", indices[0])
print("Last frame:", indices[-1])

frames = sample_video_frames(video_path)

print("Shape:", frames.shape)
print("Data type:", frames.dtype)
print("Minimum:", frames.min().item())
print("Maximum:", frames.max().item())