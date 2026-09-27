import torch

from ml.models.signspeak import SignSpeakModel
from ml.data.datasets.vocabulary import Vocabulary
from ml.training.checkpoint import load_checkpoint


def test_checkpoint_loading():
    checkpoint_path = "checkpoints/signspeak.pt"

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    vocab_size = checkpoint["vocab_size"]

    model = SignSpeakModel(
        vocab_size=vocab_size
    ).to(device)

    vocabulary = Vocabulary()

    loaded_checkpoint = load_checkpoint(
        model=model,
        optimizer=None,
        vocabulary=vocabulary,
        path=checkpoint_path,
        device=device,
    )

    model.eval()

    video = torch.randn(
        1, 16, 3, 224, 224
    ).to(device)

    decoder_input = torch.tensor(
        [[vocabulary.word_to_id["<BOS>"]]],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        logits = model(video, decoder_input)

    print("Device:", device)
    print("Restored vocabulary size:", len(vocabulary))
    print("Saved epoch:", loaded_checkpoint["epoch"])
    print("Saved loss:", loaded_checkpoint["loss"])
    print("Output shape:", logits.shape)

    assert len(vocabulary) == vocab_size
    assert logits.shape == (1, 1, vocab_size)

    print("Checkpoint loading test passed")


if __name__ == "__main__":
    test_checkpoint_loading()