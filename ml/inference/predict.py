import torch

from ml.data.datasets.vocabulary import Vocabulary
from ml.models.signspeak import SignSpeakModel
from ml.data.preprocessing.frame_sampler import sample_video_frames
from ml.training.checkpoint import load_checkpoint


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
        for _ in range(max_length):
            logits = model(video, generated_tokens)

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


def predict_video(video_path):
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model, vocabulary = load_model(
        "checkpoints/signspeak_attention_best.pt",
        device
    )

    video = sample_video_frames(
        video_path,
        num_samples=16,
        image_size=224
    )

    return greedy_decode(
        model,
        vocabulary,
        video,
        device
    )
