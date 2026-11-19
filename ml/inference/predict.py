import os
from pathlib import Path
from typing import Optional

import torch

from ml.data.datasets.vocabulary import Vocabulary
from ml.models.signspeak import SignSpeakModel
from ml.models.gesture_classifier import GestureClassifier
from ml.data.preprocessing.frame_sampler import sample_video_frames
from ml.training.checkpoint import load_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    vocabulary = Vocabulary()
    vocabulary.word_to_id = checkpoint["word_to_id"]
    vocabulary.id_to_word = checkpoint["id_to_word"]

    model = SignSpeakModel(
        vocab_size=checkpoint["vocab_size"]
    ).to(device)

    load_checkpoint(
        model=model,
        optimizer=None,
        vocabulary=vocabulary,
        path=checkpoint_path,
        device=device
    )

    model.eval()

    return model, vocabulary


def greedy_decode(model, vocabulary, video, device, max_length=50):
    video = video.unsqueeze(0).to(device)

    generated_tokens = torch.tensor(
        [[vocabulary.word_to_id["<BOS>"]]],
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():
        # Encode the video once instead of recomputing the CNN/LSTM for every
        # generated token. The decoder consumes the complete encoder sequence.
        encoder_outputs = model.encode_video(video)

        for _ in range(max_length):
            logits = model.decoder(encoder_outputs, generated_tokens)

            next_token = logits[:, -1, :].argmax(
                dim=-1,
                keepdim=True
            )

            token_id = next_token.item()

            if token_id == vocabulary.word_to_id["<EOS>"]:
                break

            generated_tokens = torch.cat(
                [generated_tokens, next_token],
                dim=1
            )

    return vocabulary.decode(
        generated_tokens[0].tolist()[1:]
    )


def get_default_checkpoint() -> Optional[str]:
    """
    Return an explicitly configured production checkpoint.

    The repository's existing *_overfit* and *_best* checkpoints were
    demonstrated to be non-generalizing or incompatible.  Selecting either
    automatically makes every upload return a memorized/empty translation.
    """
    configured = os.getenv("SIGNSPEAK_CHECKPOINT")
    if configured:
        candidate = Path(configured)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.is_file():
            return str(candidate)
        return None

    return None


def load_gesture_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    if checkpoint.get("model_type") != "gesture_classifier":
        raise RuntimeError("Checkpoint is not a gesture classifier")

    class_names = checkpoint["class_names"]
    model = GestureClassifier(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


def predict_video(video_path: str, checkpoint_path: str = None):
    if checkpoint_path is None:
        checkpoint_path = get_default_checkpoint()

    if not checkpoint_path:
        raise RuntimeError(
            "No validated SignSpeak production checkpoint is configured. "
            "Set SIGNSPEAK_CHECKPOINT to a compatible checkpoint before translating."
        )

    # Device resolution with safety fallback
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        if checkpoint.get("model_type") == "gesture_classifier":
            return predict_gesture(video_path, checkpoint_path, device)

        model, vocabulary = load_model(checkpoint_path, device)
        video = sample_video_frames(
            video_path,
            num_samples=16,
            image_size=224
        )
        return greedy_decode(model, vocabulary, video, device)
    except Exception as e:
        # Fall back to CPU if device execution fails
        if device.type != "cpu":
            device = torch.device("cpu")
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=True,
            )
            if checkpoint.get("model_type") == "gesture_classifier":
                return predict_gesture(video_path, checkpoint_path, device)

            model, vocabulary = load_model(checkpoint_path, device)
            video = sample_video_frames(
                video_path,
                num_samples=16,
                image_size=224
            )
            return greedy_decode(model, vocabulary, video, device)
        raise e


def predict_gesture(video_path, checkpoint_path, device):
    model, class_names = load_gesture_model(checkpoint_path, device)
    video = sample_video_frames(
        video_path,
        num_samples=16,
        image_size=224,
    )

    with torch.no_grad():
        logits = model(video.unsqueeze(0).to(device))
        class_id = logits.argmax(dim=-1).item()

    return class_names[class_id]
