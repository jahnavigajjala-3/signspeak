from ml.data.datasets.how2sign import How2SignDataset

dataset = How2SignDataset(
    root="data/raw/how2sign",
    split="train",
    num_samples=16,
)

print("Dataset size:", len(dataset))

video, translation = dataset[0]

print("Video shape:", video.shape)
print("Video dtype:", video.dtype)
print("Translation:", translation)