
from pathlib import Path

import torch
import torch.nn as nn

from ml.data.datasets.how2sign import How2SignDataset
from ml.data.datasets.vocabulary import Vocabulary
from ml.data.datasets.collate import make_collate_fn
from ml.models.signspeak import SignSpeakModel
from ml.training.checkpoint import save_checkpoint


def main():
    device = torch.device("cpu")
    print(f"Device: {device}")

    checkpoint_path = "checkpoints/signspeak_attention_best.pt"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    vocabulary = Vocabulary()
    vocabulary.word_to_id = checkpoint["word_to_id"]
    vocabulary.id_to_word = checkpoint["id_to_word"]

    dataset = How2SignDataset(
        root="data/raw/how2sign",
        split="val",
        num_samples=16,
        image_size=224
    )

    target_name = "-d5dN54tH2E_0-1-rgb_front.mp4"

    sample_index = next(
        (
            i for i, sample in enumerate(dataset.samples)
            if Path(sample["video_path"]).name == target_name
        ),
        None
    )

    if sample_index is None:
        raise ValueError(f"Video not found: {target_name}")

    video, translation = dataset[sample_index]

    print(f"Ground truth: {translation}")

    videos, decoder_inputs, targets = make_collate_fn(
        vocabulary
    )([(video, translation)])

    videos = videos.to(device)
    decoder_inputs = decoder_inputs.to(device)
    targets = targets.to(device)

    model = SignSpeakModel(
        vocab_size=checkpoint["vocab_size"]
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        ignore_index=vocabulary.word_to_id["<PAD>"]
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    model.train()

    for step in range(1, 301):
        optimizer.zero_grad()

        logits = model(videos, decoder_inputs)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        if step == 1 or step % 20 == 0:
            print(f"Step {step:03d} | Loss: {loss.item():.4f}")

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        vocabulary=vocabulary,
        epoch=1,
        loss=loss.item(),
        path="checkpoints/signspeak_attention_overfit.pt"
    )

    print("Attention single-video overfitting completed")


if __name__ == "__main__":
    main()
