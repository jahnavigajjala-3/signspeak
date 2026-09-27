import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int = 256,
        num_layers: int = 2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)

        return output