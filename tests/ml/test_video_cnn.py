import torch

from ml.data.preprocessing.frame_sampler import sample_video_frames
from ml.models.visual_encoder.cnn import VisualEncoder

video_path = "data/raw/how2sign/videos/train/raw_videos/DvKS20WKiX4_22-8-rgb_front.mp4"

frames = sample_video_frames(video_path)

model = VisualEncoder()

with torch.no_grad():
    features = model(frames)

print("Input shape:", frames.shape)
print("CNN output shape:", features.shape)
