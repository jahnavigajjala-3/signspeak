import torch
import torch.nn as nn


class TranslationDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        input_size: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            input_size
        )

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.context_projection = nn.Linear(
            input_size,
            num_layers * hidden_size
        )

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            vocab_size
        )

    def forward(self, encoder_outputs, target_tokens):

        context = encoder_outputs.mean(dim=1)

        hidden = self.context_projection(context)

        hidden = hidden.view(
            -1,
            self.num_layers,
            self.hidden_size
        )

        hidden = hidden.transpose(0, 1).contiguous()

        cell = torch.zeros_like(hidden)

        embedded = self.embedding(target_tokens)

        output, _ = self.lstm(
            embedded,
            (hidden, cell)
        )

        logits = self.fc(output)

        return logits
    