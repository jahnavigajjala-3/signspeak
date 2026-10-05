import re
from collections import Counter


class Vocabulary:
    def __init__(self, min_frequency: int = 2):
        self.min_frequency = min_frequency

        self.special_tokens = [
            "<PAD>",
            "<BOS>",
            "<EOS>",
            "<UNK>",
        ]

        self.word_to_id = {
            token: index
            for index, token in enumerate(self.special_tokens)
        }

        self.id_to_word = {
            index: token
            for index, token in enumerate(self.special_tokens)
        }

    def tokenize(self, sentence: str):
        sentence = sentence.lower()
        tokens = re.findall(r"\b\w+\b|[^\w\s]", sentence)
        return tokens

    def build(self, sentences):
        counter = Counter()

        for sentence in sentences:
            tokens = self.tokenize(sentence)
            counter.update(tokens)

        for word, frequency in counter.items():
            if frequency >= self.min_frequency:
                if word not in self.word_to_id:
                    index = len(self.word_to_id)
                    self.word_to_id[word] = index
                    self.id_to_word[index] = word

    def encode(self, sentence: str):
        tokens = self.tokenize(sentence)

        token_ids = [self.word_to_id["<BOS>"]]

        for token in tokens:
            token_id = self.word_to_id.get(
                token,
                self.word_to_id["<UNK>"]
            )
            token_ids.append(token_id)

        token_ids.append(self.word_to_id["<EOS>"])

        return token_ids

    def decode(self, token_ids):
        tokens = []

        for token_id in token_ids:
             word = self.id_to_word.get(token_id, "<UNK>")

             if word == "<EOS>":
               break

             if word not in ["<PAD>", "<BOS>"]:
               tokens.append(word)

        sentence = " ".join(tokens)

        sentence = re.sub(r"\s+([.,!?;:])", r"\1", sentence)
        sentence = re.sub(r"\s+(['’])\s*", r"\1", sentence)
        sentence = re.sub(r"([(\[]) ", r"\1", sentence)
        sentence = re.sub(r" ([])])", r"\1", sentence)

        return sentence
    def __len__(self):
        return len(self.word_to_id)