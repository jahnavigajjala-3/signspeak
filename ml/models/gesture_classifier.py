import torch
import torch.nn as nn

from ml.models.visual_encoder.cnn import VisualEncoder


class GestureClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        if num_classes < 2:
            raise ValueError("A gesture classifier requires at least two classes")

        self.visual_encoder = VisualEncoder()
        self.temporal_encoder = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, num_classes),
        )

    def encode_video(self, video):
        batch_size, num_frames, channels, height, width = video.shape
        flattened = video.reshape(batch_size * num_frames, channels, height, width)
        frame_features = self.visual_encoder(flattened)
        frame_features = frame_features.reshape(batch_size, num_frames, 512)
        temporal_features, _ = self.temporal_encoder(frame_features)
        return temporal_features.mean(dim=1)

    def forward(self, video):
        context = self.encode_video(video)
        return self.classifier(context)
