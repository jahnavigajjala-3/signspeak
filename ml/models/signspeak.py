import torch
import torch.nn as nn

from ml.models.visual_encoder.cnn import VisualEncoder
from ml.models.temporal_encoder.lstm import TemporalEncoder
from ml.models.translation.decoder import TranslationDecoder


class SignSpeakModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        visual_feature_size: int = 512,
        temporal_hidden_size: int = 256,
        decoder_hidden_size: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()

        self.visual_encoder = VisualEncoder()

        self.temporal_encoder = TemporalEncoder(
            input_size=visual_feature_size,
            hidden_size=temporal_hidden_size,
            num_layers=num_layers,
        )

        self.decoder = TranslationDecoder(
            vocab_size=vocab_size,
            input_size=temporal_hidden_size,
            hidden_size=decoder_hidden_size,
            num_layers=num_layers,
        )

    def encode_video(self, video):
        batch_size, num_frames, channels, height, width = video.shape

        video = video.reshape(
            batch_size * num_frames,
            channels,
            height,
            width,
        )

        visual_features = self.visual_encoder(video)

        visual_features = visual_features.reshape(
            batch_size,
            num_frames,
            -1,
        )

        return self.temporal_encoder(visual_features)

    def forward(self, video, target_tokens, teacher_forcing_ratio: float = 1.0):
        encoder_outputs = self.encode_video(video)

        logits = self.decoder(
            encoder_outputs,
            target_tokens,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        return logits
    