from ml.data.datasets.how2sign import How2SignDataset
from ml.data.datasets.vocabulary import Vocabulary

dataset = How2SignDataset(
    root="data/raw/how2sign",
    split="train",
)

sentences = [
    sample["translation"]
    for sample in dataset.samples
]

vocab = Vocabulary(min_frequency=2)
vocab.build(sentences)

sentence = "I want some water."

encoded = vocab.encode(sentence)
decoded = vocab.decode(encoded)

print("Vocabulary size:", len(vocab))
print("Original:", sentence)
print("Encoded:", encoded)
print("Decoded:", decoded)

assert encoded[0] == vocab.word_to_id["<BOS>"]
assert encoded[-1] == vocab.word_to_id["<EOS>"]
assert decoded == sentence.lower()