import torch
import torch.nn as nn
import torch.nn.functional as F


class TranslationDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        input_size: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size

        self.embedding = nn.Embedding(
            vocab_size,
            input_size
        )

        self.context_projection = nn.Linear(
            input_size,
            num_layers * hidden_size
        )

        self.attention_encoder = nn.Linear(
            input_size,
            hidden_size,
            bias=False
        )

        self.attention_decoder = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False
        )

        self.attention_score = nn.Linear(
            hidden_size,
            1,
            bias=False
        )

        self.lstm = nn.LSTM(
            input_size=input_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            vocab_size
        )

    def forward(
        self,
        encoder_outputs,
        target_tokens,
        teacher_forcing_ratio: float = 1.0,
    ):

        batch_size = encoder_outputs.size(0)

        context = encoder_outputs.mean(dim=1)

        hidden = self.context_projection(context)

        hidden = hidden.view(
            batch_size,
            self.num_layers,
            self.hidden_size
        )

        hidden = hidden.transpose(0, 1).contiguous()

        cell = torch.zeros_like(hidden)

        encoder_features = self.attention_encoder(
            encoder_outputs
        )

        # First decoder input is always the ground-truth BOS token.
        input_tokens = target_tokens[:, 0]

        outputs = []

        for step in range(target_tokens.size(1)):

            decoder_hidden = hidden[-1].unsqueeze(1)

            attention_energy = torch.tanh(
                encoder_features
                + self.attention_decoder(decoder_hidden)
            )

            attention_scores = self.attention_score(
                attention_energy
            ).squeeze(-1)

            attention_weights = F.softmax(
                attention_scores,
                dim=-1
            )

            attended_context = torch.bmm(
                attention_weights.unsqueeze(1),
                encoder_outputs
            )

            current_embedding = self.embedding(
                input_tokens
            ).unsqueeze(1)

            lstm_input = torch.cat(
                [
                    current_embedding,
                    attended_context
                ],
                dim=-1
            )

            output, (hidden, cell) = self.lstm(
                lstm_input,
                (hidden, cell)
            )

            logits = self.fc(output)

            outputs.append(logits)

            # Scheduled sampling: after the first step, occasionally feed the
            # model's own prediction instead of the ground-truth token. This
            # exposes the decoder to its own errors and forces the attention
            # context (i.e. the video) to carry sentence-identifying signal.
            if step + 1 < target_tokens.size(1):
                if teacher_forcing_ratio >= 1.0:
                    input_tokens = target_tokens[:, step + 1]
                elif teacher_forcing_ratio <= 0.0:
                    input_tokens = logits[:, -1, :].argmax(
                        dim=-1
                    ).detach()
                elif torch.rand(1).item() < teacher_forcing_ratio:
                    input_tokens = target_tokens[:, step + 1]
                else:
                    input_tokens = logits[:, -1, :].argmax(
                        dim=-1
                    ).detach()

        return torch.cat(outputs, dim=1)