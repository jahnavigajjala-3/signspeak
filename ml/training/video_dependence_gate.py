"""Tiny video-dependence gate.

The task is deliberately unsolvable from text history alone:

    video_i  ->  <BOS> word_i <EOS>

The decoder's first input is always <BOS>, which is identical for every
video. Therefore the first predicted token can only be correct if the
model conditions on the video (through the attention context).

Pass criteria:
  a. |TF loss(real video) - TF loss(zero video)| >= 1.0
  b. Greedy decoding returns the correct distinct token for every video
  c. With the video->target assignment shuffled, greedy accuracy drops to
     chance (1/n) or below

Variants (for bisection if the baseline fails):
  baseline   : current custom CNN + LSTM + attention decoder
  layernorm  : LayerNorm on the CNN features
  high_lr    : higher learning rate (3e-3), no gradient clipping change
  resnet     : pretrained ResNet18 backbone instead of the custom CNN

Run:
    python -m ml.training.video_dependence_gate
    python -m ml.training.video_dependence_gate --variant layernorm
    python -m ml.training.video_dependence_gate --variant resnet
"""

import argparse
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ml.data.datasets.how2sign import How2SignDataset
from ml.data.datasets.vocabulary import Vocabulary
from ml.models.visual_encoder.cnn import VisualEncoder
from ml.models.temporal_encoder.lstm import TemporalEncoder
from ml.models.translation.decoder import TranslationDecoder
from ml.inference.predict import greedy_decode

SEED = 42
NUM_VIDEOS = 5
TARGET_WORDS = ["water", "apple", "music", "happy", "friend"]
DEFAULT_STEPS = 500
BASE_LR = 1e-3
HIGH_LR = 3e-3
CHECKPOINT_DIR = "checkpoints"
VOCAB_CHECKPOINT = "checkpoints/signspeak_attention_best.pt"
_SENTENCE_NAME_RE = re.compile(r"^(.+)_\d+-\d+-rgb_front$")


class ResNetVisualEncoder(nn.Module):
    """Pretrained ResNet18 producing one 512-d feature per frame."""

    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights

        self.pretrained = False
        try:
            net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.pretrained = True
        except Exception as exc:
            print(f"Could not download pretrained weights ({exc}); using random ResNet")
            net = resnet18(weights=None)

        net.fc = nn.Identity()
        self.net = net
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.net(x)


class GateModel(nn.Module):
    """Same encoder/decoder wiring as SignSpeakModel, with optional variants."""

    def __init__(self, vocab_size, variant="baseline"):
        super().__init__()
        self.variant = variant

        if variant in ("resnet", "resnet_frozen"):
            self.visual_encoder = ResNetVisualEncoder()
        else:
            self.visual_encoder = VisualEncoder()

        self.feature_norm = (
            nn.LayerNorm(512) if variant == "layernorm" else nn.Identity()
        )
        self.temporal_encoder = TemporalEncoder(
            input_size=512, hidden_size=256, num_layers=2
        )
        self.output_norm = (
            nn.LayerNorm(256) if variant == "encoder_layernorm"
            else nn.Identity()
        )
        self.decoder = TranslationDecoder(
            vocab_size=vocab_size,
            input_size=256,
            hidden_size=256,
            num_layers=2,
        )

        if variant == "freeze_encoder":
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
            for param in self.temporal_encoder.parameters():
                param.requires_grad = False

        if variant == "resnet_frozen":
            for param in self.visual_encoder.parameters():
                param.requires_grad = False

    def encode_video(self, video):
        batch_size, num_frames, channels, height, width = video.shape
        video = video.reshape(
            batch_size * num_frames, channels, height, width
        )
        features = self.visual_encoder(video)
        features = features.reshape(batch_size, num_frames, -1)
        features = self.feature_norm(features)
        encoded = self.temporal_encoder(features)
        return self.output_norm(encoded)

    def forward(self, video, target_tokens, teacher_forcing_ratio=1.0):
        return self.decoder(
            self.encode_video(video),
            target_tokens,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_vocabulary():
    if not os.path.exists(VOCAB_CHECKPOINT):
        raise FileNotFoundError(
            f"Vocabulary checkpoint not found: {VOCAB_CHECKPOINT}"
        )
    checkpoint = torch.load(
        VOCAB_CHECKPOINT, map_location="cpu", weights_only=True
    )
    vocabulary = Vocabulary()
    vocabulary.word_to_id = checkpoint["word_to_id"]
    vocabulary.id_to_word = checkpoint["id_to_word"]
    return vocabulary


def select_videos(dataset, count):
    selected = []
    seen_ids = set()
    for sample in dataset.samples:
        stem = Path(sample["video_path"]).stem
        match = _SENTENCE_NAME_RE.match(stem)
        video_id = match.group(1) if match else stem
        if video_id in seen_ids:
            continue
        seen_ids.add(video_id)
        selected.append(sample)
        if len(selected) == count:
            break
    return selected


def build_gate_batch(dataset, vocabulary, samples):
    videos = []
    for sample in samples:
        video, _ = dataset[dataset.samples.index(sample)]
        videos.append(video)
    videos = torch.stack(videos)

    bos_id = vocabulary.word_to_id["<BOS>"]
    eos_id = vocabulary.word_to_id["<EOS>"]

    decoder_inputs = torch.tensor(
        [[bos_id, vocabulary.word_to_id[word]] for word in TARGET_WORDS],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [[vocabulary.word_to_id[word], eos_id] for word in TARGET_WORDS],
        dtype=torch.long,
    )
    return videos, decoder_inputs, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        default="baseline",
        choices=[
            "baseline",
            "layernorm",
            "high_lr",
            "resnet",
            "freeze_encoder",
            "encoder_layernorm",
            "low_lr",
        ],
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    args = parser.parse_args()

    set_seed()
    device = torch.device("cpu")
    variant = args.variant

    print(f"Variant: {variant}")
    print(f"Device: {device}")

    dataset = How2SignDataset(
        root="data/raw/how2sign",
        split="train",
        num_samples=16,
        image_size=224,
    )
    vocabulary = load_vocabulary()

    samples = select_videos(dataset, NUM_VIDEOS)
    print("\nSelected videos:")
    for index, sample in enumerate(samples):
        print(
            f"  [{index}] {Path(sample['video_path']).name} "
            f"-> \"{TARGET_WORDS[index]}\""
        )

    videos, decoder_inputs, targets = build_gate_batch(
        dataset, vocabulary, samples
    )
    print(f"videos: {tuple(videos.shape)}")
    print(f"decoder_inputs: {decoder_inputs.tolist()}")
    print(f"targets: {targets.tolist()}")

    model = GateModel(
        vocab_size=len(vocabulary), variant=variant
    ).to(device)

    if variant == "resnet":
        print(
            "ResNet pretrained weights: "
            f"{model.visual_encoder.pretrained}"
        )

    pad_id = vocabulary.word_to_id["<PAD>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    learning_rate = BASE_LR
    if variant == "high_lr":
        learning_rate = HIGH_LR
    elif variant == "low_lr":
        learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\nTraining (lr={learning_rate}, max_steps={args.steps})...")
    model.train()
    final_loss = None
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        logits = model(
            videos, decoder_inputs, teacher_forcing_ratio=1.0
        )
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        final_loss = loss.item()

        if step == 1 or step % 25 == 0:
            print(f"  step {step:04d} | loss {final_loss:.4f}")
        if final_loss < 0.02:
            print(f"  early stop at step {step} (loss {final_loss:.4f})")
            break

    # ==================================================
    # Evaluation
    # ==================================================
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        logits_real = model(videos, decoder_inputs)
        logits_zero = model(torch.zeros_like(videos), decoder_inputs)
        loss_real = criterion(
            logits_real.reshape(-1, logits_real.size(-1)),
            targets.reshape(-1),
        ).item()
        loss_zero = criterion(
            logits_zero.reshape(-1, logits_zero.size(-1)),
            targets.reshape(-1),
        ).item()

    print(f"TF loss real video: {loss_real:.4f}")
    print(f"TF loss zero video: {loss_zero:.4f}")
    print(f"loss gap: {abs(loss_real - loss_zero):.4f}")

    # Greedy decoding with the correct assignment.
    predictions = []
    for index in range(NUM_VIDEOS):
        prediction = greedy_decode(
            model, vocabulary, videos[index], device, max_length=5
        )
        tokens = vocabulary.tokenize(prediction)
        predictions.append(tokens[0] if tokens else "")

    print("\nGreedy predictions (correct assignment):")
    correct = 0
    for index, prediction in enumerate(predictions):
        is_correct = prediction == TARGET_WORDS[index]
        correct += int(is_correct)
        print(
            f"  video {index}: target=\"{TARGET_WORDS[index]}\" "
            f"pred=\"{prediction}\" {'OK' if is_correct else 'WRONG'}"
        )

    # Shuffled assignment: video i is checked against target (i+1) % n.
    shuffled_correct = 0
    print("\nGreedy predictions (shuffled assignment):")
    for index in range(NUM_VIDEOS):
        shuffled_target = TARGET_WORDS[(index + 1) % NUM_VIDEOS]
        is_correct = predictions[index] == shuffled_target
        shuffled_correct += int(is_correct)
        print(
            f"  video {index}: shuffled_target=\"{shuffled_target}\" "
            f"pred=\"{predictions[index]}\" "
            f"{'OK' if is_correct else 'WRONG'}"
        )

    accuracy = correct / NUM_VIDEOS
    shuffled_accuracy = shuffled_correct / NUM_VIDEOS
    chance = 1.0 / NUM_VIDEOS

    criterion_a = abs(loss_real - loss_zero) >= 1.0
    criterion_b = correct == NUM_VIDEOS
    criterion_c = shuffled_accuracy <= chance

    print("\n" + "=" * 60)
    print("GATE CRITERIA")
    print("=" * 60)
    print(
        f"a. loss gap {abs(loss_real - loss_zero):.4f} >= 1.0: "
        f"{'PASS' if criterion_a else 'FAIL'}"
    )
    print(
        f"b. greedy correct {correct}/{NUM_VIDEOS}: "
        f"{'PASS' if criterion_b else 'FAIL'}"
    )
    print(
        f"c. shuffled accuracy {shuffled_accuracy:.2f} <= chance "
        f"{chance:.2f}: {'PASS' if criterion_c else 'FAIL'}"
    )
    print(
        f"\nOVERALL: {'PASS' if (criterion_a and criterion_b and criterion_c) else 'FAIL'}"
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(
        CHECKPOINT_DIR, f"video_dependence_gate_{variant}.pt"
    )
    torch.save(
        {
            "variant": variant,
            "loss": final_loss,
            "loss_real": loss_real,
            "loss_zero": loss_zero,
            "model_state_dict": model.state_dict(),
            "word_to_id": vocabulary.word_to_id,
            "id_to_word": vocabulary.id_to_word,
            "vocab_size": len(vocabulary),
            "target_words": TARGET_WORDS,
        },
        checkpoint_path,
    )
    print(f"\nSaved gate checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
