import torch

from ml.data.preprocessing.frame_sampler import sample_video_frames
from ml.models.visual_encoder.cnn import VisualEncoder
from ml.models.temporal_encoder.lstm import TemporalEncoder

video_path = "data/raw/how2sign/videos/train/raw_videos/DvKS20WKiX4_22-8-rgb_front.mp4"

frames = sample_video_frames(video_path)

visual_encoder = VisualEncoder()
temporal_encoder = TemporalEncoder()

with torch.no_grad():
    visual_features = visual_encoder(frames)

    visual_features = visual_features.unsqueeze(0)

    temporal_features = temporal_encoder(visual_features)

print("Frames:", frames.shape)
print("Visual features:", visual_features.shape)
print("Temporal features:", temporal_features.shape)