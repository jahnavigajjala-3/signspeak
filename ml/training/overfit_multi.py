"""Controlled multi-video overfitting experiment.

Purpose: establish whether the current architecture and training pipeline
(video -> CNN -> LSTM -> attention decoder) can memorise several distinct
How2Sign examples, and whether greedy decoding becomes video-dependent.

This trains a FRESH model from random initialisation on CPU. It does not
touch the production one-video checkpoint.
"""

import os
import random
import re
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ml.data.datasets.how2sign import How2SignDataset
from ml.data.datasets.vocabulary import Vocabulary
from ml.data.datasets.collate import make_collate_fn
from ml.models.signspeak import SignSpeakModel
from ml.inference.predict import greedy_decode
from ml.training.checkpoint import save_checkpoint
from torch.nn.utils.rnn import pad_sequence

SEED = 42
NUM_EXAMPLES = 10
NUM_STEPS = 400
BATCH_SIZE = 4
LEARNING_RATE = 5e-4
TF_START = 1.0
TF_END = 0.4
CHECKPOINT_PATH = "checkpoints/signspeak_attention_multi_overfit.pt"
VOCAB_CHECKPOINT = "checkpoints/signspeak_attention_best.pt"

_SENTENCE_NAME_RE = re.compile(r"^(.+)_\d+-\d+-rgb_front$")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def video_id_for(sample):
    """Group sentence clips by the underlying How2Sign video ID."""
    stem = Path(sample["video_path"]).stem
    match = _SENTENCE_NAME_RE.match(stem)
    return match.group(1) if match else stem


def token_count(text):
    return len(re.findall(r"\b\w+\b|[^\w\s]", text))


def select_examples(dataset, count):
    """Deterministic selection of distinct videos with distinct translations."""
    candidates = []
    seen_video_ids = set()
    seen_translations = set()

    for sample in dataset.samples:
        text = sample["translation"].strip()
        if not text or token_count(text) < 4:
            continue

        vid = video_id_for(sample)
        normalised_text = text.lower()

        if vid in seen_video_ids or normalised_text in seen_translations:
            continue

        seen_video_ids.add(vid)
        seen_translations.add(normalised_text)
        candidates.append(sample)

    rng = random.Random(SEED)
    rng.shuffle(candidates)

    if len(candidates) < count:
        raise RuntimeError(
            f"Only {len(candidates)} distinct candidates available, "
            f"needed {count}"
        )

    return candidates[:count]


def load_vocabulary(dataset):
    """Use the existing production vocabulary when available."""
    if os.path.exists(VOCAB_CHECKPOINT):
        checkpoint = torch.load(
            VOCAB_CHECKPOINT,
            map_location="cpu",
            weights_only=True,
        )
        vocabulary = Vocabulary()
        vocabulary.word_to_id = checkpoint["word_to_id"]
        vocabulary.id_to_word = checkpoint["id_to_word"]
        print(f"Loaded existing vocabulary from {VOCAB_CHECKPOINT}")
        return vocabulary

    print(
        f"Vocabulary checkpoint {VOCAB_CHECKPOINT} not found; "
        "building from train translations (min_frequency=2)"
    )
    vocabulary = Vocabulary(min_frequency=2)
    vocabulary.build(sample["translation"] for sample in dataset.samples)
    return vocabulary


def teacher_forced_accuracy(logits, targets, pad_id):
    predictions = logits.argmax(dim=-1)
    mask = targets != pad_id
    total = mask.sum().item()
    if total == 0:
        return 0.0
    correct = (predictions == targets) & mask
    return correct.sum().item() / total * 100.0


def main():
    set_seed()
    device = torch.device("cpu")
    print(f"Device: {device}")
    print("Initializing FRESH SignSpeakModel (random initialization)...")

    dataset = How2SignDataset(
        root="data/raw/how2sign",
        split="train",
        num_samples=16,
        image_size=224,
    )

    selected_samples = select_examples(dataset, NUM_EXAMPLES)
    sample_indices = [
        dataset.samples.index(sample)
        for sample in selected_samples
    ]

    print("\n==================================================")
    print(f"SELECTED TRAINING EXAMPLES (seed={SEED})")
    print("==================================================")
    for idx, sample in enumerate(selected_samples):
        print(f"[{idx + 1:02d}] Video: {Path(sample['video_path']).name}")
        print(f"     Video ID: {video_id_for(sample)}")
        print(f"     GT:       \"{sample['translation']}\"\n")

    vocabulary = load_vocabulary(dataset)
    pad_id = vocabulary.word_to_id["<PAD>"]
    bos_id = vocabulary.word_to_id["<BOS>"]
    eos_id = vocabulary.word_to_id["<EOS>"]

    print("==================================================")
    print("VOCABULARY DIAGNOSTICS")
    print("==================================================")
    print(f"Vocab Size: {len(vocabulary)}")
    print(
        f"Special Tokens: <PAD>={pad_id}, <BOS>={bos_id}, "
        f"<EOS>={eos_id}, <UNK>={vocabulary.word_to_id['<UNK>']}\n"
    )

    model = SignSpeakModel(vocab_size=len(vocabulary)).to(device)
    collate_fn = make_collate_fn(vocabulary)

    print("Pre-loading frame tensors for training samples...")
    processed_data = []
    for sample_index, sample in zip(sample_indices, selected_samples):
        video_tensor, text = dataset[sample_index]
        videos, decoder_inputs, targets = collate_fn([(video_tensor, text)])
        processed_data.append({
            "sample": sample,
            "video_tensor": video_tensor,
            "video_batch": videos,
            "decoder_inputs": decoder_inputs.squeeze(0),
            "targets": targets.squeeze(0),
            "text": text,
        })

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n==================================================")
    print("TRAINING LOOP (CPU, scheduled sampling)")
    print("==================================================")

    model.train()
    initial_loss = None
    final_loss = None
    order = list(range(len(processed_data)))
    rng = random.Random(SEED)

    for step in range(1, NUM_STEPS + 1):
        progress = (step - 1) / max(1, NUM_STEPS - 1)
        teacher_forcing_ratio = TF_START + (TF_END - TF_START) * progress

        rng.shuffle(order)
        optimizer.zero_grad()
        step_loss = 0.0

        for start in range(0, len(order), BATCH_SIZE):
            batch_indices = order[start:start + BATCH_SIZE]
            batch = [processed_data[i] for i in batch_indices]

            videos = torch.cat(
                [item["video_batch"] for item in batch], dim=0
            )
            decoder_inputs = pad_sequence(
                [item["decoder_inputs"] for item in batch],
                batch_first=True,
                padding_value=pad_id,
            )
            targets = pad_sequence(
                [item["targets"] for item in batch],
                batch_first=True,
                padding_value=pad_id,
            )

            logits = model(
                videos,
                decoder_inputs,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

            loss.backward()
            step_loss += loss.item() * len(batch)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        average_loss = step_loss / len(processed_data)

        if step == 1:
            initial_loss = average_loss
            print(
                f"Step {step:03d} | Initial Loss: {average_loss:.4f} "
                f"| teacher_forcing={teacher_forcing_ratio:.2f}"
            )
        elif step % 25 == 0 or step == NUM_STEPS:
            print(
                f"Step {step:03d} | Loss: {average_loss:.4f} "
                f"| teacher_forcing={teacher_forcing_ratio:.2f}"
            )
            final_loss = average_loss

    os.makedirs("checkpoints", exist_ok=True)
    if os.path.exists(CHECKPOINT_PATH):
        v1_path = CHECKPOINT_PATH.replace(
            ".pt", "_v1.pt"
        )
        if not os.path.exists(v1_path):
            shutil.copy2(CHECKPOINT_PATH, v1_path)
            print(f"Preserved previous experiment checkpoint: {v1_path}")

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        vocabulary=vocabulary,
        epoch=NUM_STEPS,
        loss=final_loss,
        path=CHECKPOINT_PATH,
    )
    print(f"Saved multi-video overfit checkpoint to '{CHECKPOINT_PATH}'")

    # ==================================================
    # EVALUATION & DIAGNOSTICS
    # ==================================================
    print("\n==================================================")
    print("EVALUATION & DIAGNOSTICS")
    print("==================================================")

    model.eval()
    accuracies = []
    greedy_predictions = []
    real_losses = []
    zero_losses = []

    for item in processed_data:
        video_tensor = item["video_tensor"]
        video_batch = item["video_batch"]
        decoder_inputs = item["decoder_inputs"]
        targets = item["targets"]
        ground_truth = item["text"]
        video_name = Path(item["sample"]["video_path"]).name

        with torch.no_grad():
            logits = model(
                video_batch, decoder_inputs.unsqueeze(0)
            )
            accuracy = teacher_forced_accuracy(
                logits, targets.unsqueeze(0), pad_id
            )
            accuracies.append(accuracy)

            real_loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            ).item()
            real_losses.append(real_loss)

            zero_logits = model(
                torch.zeros_like(video_batch),
                decoder_inputs.unsqueeze(0),
            )
            zero_loss = criterion(
                zero_logits.reshape(-1, zero_logits.size(-1)),
                targets.reshape(-1),
            ).item()
            zero_losses.append(zero_loss)

            prediction = greedy_decode(
                model, vocabulary, video_tensor, device
            )
            greedy_predictions.append(prediction)

        print(f"VIDEO: {video_name}")
        print(f"GROUND TRUTH: \"{ground_truth}\"")
        print(f"GREEDY PREDICTION: \"{prediction}\"")
        print(f"TEACHER-FORCED TOKEN ACCURACY: {accuracy:.2f}%")
        print(
            f"TF LOSS real/zero video: {real_loss:.4f} / {zero_loss:.4f}\n"
        )

    average_accuracy = sum(accuracies) / len(accuracies)
    average_real_loss = sum(real_losses) / len(real_losses)
    average_zero_loss = sum(zero_losses) / len(zero_losses)

    print("==================================================")
    print("SUMMARY METRICS")
    print("==================================================")
    print(f"Number of Examples: {len(processed_data)}")
    print(f"Initial Loss: {initial_loss:.4f}")
    print(f"Final Loss:   {final_loss:.4f}")
    print(f"Average Teacher-Forced Accuracy: {average_accuracy:.2f}%")
    print(
        f"Average TF Loss real video: {average_real_loss:.4f} "
        f"| zero video: {average_zero_loss:.4f}"
    )
    print(
        f"Unique Greedy Predictions: "
        f"{len(set(greedy_predictions))} / {len(processed_data)}"
    )
    print("==================================================")


if __name__ == "__main__":
    main()
