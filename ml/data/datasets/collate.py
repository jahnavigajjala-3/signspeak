import torch
from torch.nn.utils.rnn import pad_sequence


def make_collate_fn(vocabulary):
    pad_id = vocabulary.word_to_id["<PAD>"]

    def collate_fn(batch):
        videos = []
        decoder_inputs = []
        targets = []

        for video, translation in batch:
            token_ids = vocabulary.encode(translation)

            input_tokens = token_ids[:-1]
            target_tokens = token_ids[1:]

            videos.append(video)
            decoder_inputs.append(
                torch.tensor(input_tokens, dtype=torch.long)
            )
            targets.append(
                torch.tensor(target_tokens, dtype=torch.long)
            )

        videos = torch.stack(videos)

        decoder_inputs = pad_sequence(
            decoder_inputs,
            batch_first=True,
            padding_value=pad_id,
        )

        targets = pad_sequence(
            targets,
            batch_first=True,
            padding_value=pad_id,
        )

        return videos, decoder_inputs, targets

    return collate_fn