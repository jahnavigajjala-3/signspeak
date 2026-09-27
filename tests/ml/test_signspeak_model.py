import torch

from ml.data.preprocessing.frame_sampler import sample_video_frames
from ml.models.visual_encoder.cnn import VisualEncoder
from ml.models.temporal_encoder.lstm import TemporalEncoder
from ml.models.translation.decoder import TranslationDecoder


video_path = "data/raw/how2sign/videos/train/raw_videos/DvKS20WKiX4_22-8-rgb_front.mp4"

vocab_size = 1000
target_length = 8

frames = sample_video_frames(video_path)

visual_encoder = VisualEncoder()
temporal_encoder = TemporalEncoder()
decoder = TranslationDecoder(vocab_size=vocab_size)

with torch.no_grad():
    visual_features = visual_encoder(frames)

    visual_features = visual_features.unsqueeze(0)

    temporal_features = temporal_encoder(visual_features)

    target_tokens = torch.randint(
        0,
        vocab_size,
        (1, target_length)
    )

    logits = decoder(
        temporal_features,
        target_tokens
    )

print("Frames:", frames.shape)
print("Visual features:", visual_features.shape)
print("Temporal features:", temporal_features.shape)
print("Target tokens:", target_tokens.shape)
print("Decoder logits:", logits.shape)
