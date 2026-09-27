import torch

from ml.models.signspeak import SignSpeakModel


def test_signspeak_model():
    batch_size = 1
    num_frames = 16
    vocab_size = 9866
    target_length = 7

    model = SignSpeakModel(
        vocab_size=vocab_size
    )

    video = torch.randn(
        batch_size,
        num_frames,
        3,
        224,
        224,
    )

    target_tokens = torch.randint(
        0,
        vocab_size,
        (batch_size, target_length),
    )

    logits = model(video, target_tokens)

    assert logits.shape == (
        batch_size,
        target_length,
        vocab_size,
    )

    print("Output shape:", logits.shape)
    print("SignSpeak integration test passed")


if __name__ == "__main__":
    test_signspeak_model()