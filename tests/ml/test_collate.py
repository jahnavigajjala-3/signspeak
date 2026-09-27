from torch.utils.data import DataLoader

from ml.data.datasets.how2sign import How2SignDataset
from ml.data.datasets.vocabulary import Vocabulary
from ml.data.datasets.collate import make_collate_fn


def test_collate():
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

    collate_fn = make_collate_fn(vocabulary)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    videos, decoder_inputs, targets = next(iter(loader))

    print("Videos:", videos.shape)
    print("Decoder inputs:", decoder_inputs.shape)
    print("Targets:", targets.shape)

    print("Video dtype:", videos.dtype)
    print("Token dtype:", decoder_inputs.dtype)

    print("First decoder input:", decoder_inputs[0])
    print("First target:", targets[0])

    assert videos.shape == (2, 16, 3, 224, 224)
    assert decoder_inputs.shape == targets.shape
    assert decoder_inputs.dtype.is_floating_point is False
    assert targets.dtype.is_floating_point is False

    print("Collator test passed")


if __name__ == "__main__":
    test_collate()