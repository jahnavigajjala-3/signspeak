import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.data.datasets.how2sign import How2SignDataset
from ml.data.datasets.vocabulary import Vocabulary
from ml.data.datasets.collate import make_collate_fn
from ml.models.signspeak import SignSpeakModel
from ml.training.checkpoint import save_checkpoint

def train_smoke_test():
    dataset = How2SignDataset(
        root="data/raw/how2sign",
        split="train",
        num_samples=16,
        image_size=224,
    )

    vocabulary = Vocabulary(min_frequency=2)
    vocabulary.build(
        sample["translation"]
        for sample in dataset.samples
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=make_collate_fn(vocabulary),
    )

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
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

    model.train()

    videos, decoder_inputs, targets = next(iter(loader))

    videos = videos.to(device)
    decoder_inputs = decoder_inputs.to(device)
    targets = targets.to(device)

    for step in range(1, 21):
        optimizer.zero_grad()

        logits = model(videos, decoder_inputs)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

        loss.backward()
        optimizer.step()

        print(f"Step {step:02d} | Loss: {loss.item():.4f}")

    print("Tiny-batch overfitting experiment completed")
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        vocabulary=vocabulary,
        epoch=1,
        loss=loss.item(),
    )

if __name__ == "__main__":
    train_smoke_test()