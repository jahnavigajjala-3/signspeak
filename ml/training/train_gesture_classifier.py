import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

from ml.data.datasets.gestures import GestureDataset
from ml.models.gesture_classifier import GestureClassifier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def validate_video_dependence(model, dataset, device):
    model.eval()
    video, class_id = dataset[0]
    video = video.unsqueeze(0).to(device)
    zero_video = torch.zeros_like(video)

    with torch.no_grad():
        real_logits = model(video)
        zero_logits = model(zero_video)

    max_diff = (real_logits - zero_logits).abs().max().item()
    real_prediction = real_logits.argmax(dim=-1).item()
    zero_prediction = zero_logits.argmax(dim=-1).item()

    return {
        "real_prediction": dataset.class_names[real_prediction],
        "zero_prediction": dataset.class_names[zero_prediction],
        "max_logit_diff": max_diff,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Folder containing one subfolder per gesture")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="checkpoints/signspeak_gesture.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    dataset = GestureDataset(args.data)
    if len(dataset) < 10:
        raise ValueError("Provide at least 10 gesture videos for a holdout split")

    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = GestureClassifier(num_classes=len(dataset.class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for videos, labels in train_loader:
            videos = videos.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(videos)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for videos, labels in val_loader:
                videos = videos.to(device)
                labels = labels.to(device)
                predictions = model(videos).argmax(dim=-1)
                correct += (predictions == labels).sum().item()
                total += len(labels)

        accuracy = correct / total if total else 0.0
        print(f"Epoch {epoch:03d} loss={total_loss / train_size:.4f} val_accuracy={accuracy:.4f}")

    diagnostics = validate_video_dependence(model, dataset, device)
    print("Video-dependence:", diagnostics)

    if diagnostics["real_prediction"] == diagnostics["zero_prediction"]:
        print("WARNING: real and zero-video predictions match; checkpoint is not video-dependent.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "gesture_classifier",
            "model_state_dict": model.state_dict(),
            "class_names": dataset.class_names,
            "num_classes": len(dataset.class_names),
            "validation_diagnostics": diagnostics,
        },
        output,
    )
    print(f"Saved checkpoint: {output}")


if __name__ == "__main__":
    main()
