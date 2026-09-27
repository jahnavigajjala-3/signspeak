from pathlib import Path

import torch


def save_checkpoint(
    model,
    optimizer,
    vocabulary,
    epoch: int,
    loss: float,
    path: str = "checkpoints/signspeak.pt",
):
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "word_to_id": vocabulary.word_to_id,
        "id_to_word": vocabulary.id_to_word,
        "vocab_size": len(vocabulary),
    }

    torch.save(checkpoint, checkpoint_path)

    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(
    model,
    optimizer,
    vocabulary,
    path="checkpoints/signspeak.pt",
    device="cpu",
):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    vocabulary.word_to_id = checkpoint["word_to_id"]
    vocabulary.id_to_word = checkpoint["id_to_word"]

    return checkpoint