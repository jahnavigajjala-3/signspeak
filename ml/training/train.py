import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.data.datasets.how2sign import How2SignDataset
from ml.data.datasets.vocabulary import Vocabulary
from ml.data.datasets.collate import make_collate_fn
from ml.models.signspeak import SignSpeakModel
from ml.training.checkpoint import save_checkpoint

def get_device():
    return torch.device("cpu")


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    max_batches=None,
):
    model.train()

    total_loss = 0.0
    num_batches = 0

    for batch_index, (videos, decoder_inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        videos = videos.to(device)
        decoder_inputs = decoder_inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(videos, decoder_inputs)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        print(
            f"Train batch {batch_index + 1} | "
            f"Loss: {loss.item():.4f}"
        )

    return total_loss / max(num_batches, 1)


def validate(
    model,
    loader,
    criterion,
    device,
    max_batches=None,
):
    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_index, (videos, decoder_inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            videos = videos.to(device)
            decoder_inputs = decoder_inputs.to(device)
            targets = targets.to(device)

            logits = model(videos, decoder_inputs)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

            total_loss += loss.item()
            num_batches += 1

            print(
                f"Validation batch {batch_index + 1} | "
                f"Loss: {loss.item():.4f}"
            )

    return total_loss / max(num_batches, 1)


def train():
    device = get_device()
    print("Device:", device)

    train_dataset = How2SignDataset(
        root="data/raw/how2sign",
        split="train",
        num_samples=16,
        image_size=224,
    )

    val_dataset = How2SignDataset(
        root="data/raw/how2sign",
        split="val",
        num_samples=16,
        image_size=224,
    )

    vocabulary = Vocabulary(min_frequency=2)
    vocabulary.build(
        sample["translation"]
        for sample in train_dataset.samples
    )

    collate_fn = make_collate_fn(vocabulary)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = SignSpeakModel(
        vocab_size=len(vocabulary)
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        ignore_index=vocabulary.word_to_id["<PAD>"]
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    epochs = 3
    max_train_batches = 20
    max_val_batches = 5

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_batches=max_train_batches,
        )

        val_loss = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=max_val_batches,
        )

        print(f"Epoch {epoch} training loss: {train_loss:.4f}")
        print(f"Epoch {epoch} validation loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                vocabulary=vocabulary,
                epoch=epoch,
                loss=val_loss,
                path="checkpoints/signspeak_attention_best.pt",
            )

            print("New best validation checkpoint saved")

    print("\nTraining completed")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()