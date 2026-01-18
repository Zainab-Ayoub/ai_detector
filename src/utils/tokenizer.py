import json
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.utils.helpers import clean_text


class TextTokenizer:
    def __init__(self, max_vocab=20000, max_len=300):
        """
        max_vocab = size of vocabulary (changed from num_words)
        max_len = maximum sequence length for padding
        """
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
        self.word_index = {}

    def fit(self, texts):
        """
        Fit tokenizer on cleaned text corpus
        """
        cleaned = [clean_text(t) for t in texts]
        self.tokenizer.fit_on_texts(cleaned)
        self.word_index = self.tokenizer.word_index

    def texts_to_sequences(self, texts):
        """
        Convert raw text -> padded integer sequences
        """
        cleaned = [clean_text(t) for t in texts]
        seq = self.tokenizer.texts_to_sequences(cleaned)
        padded = pad_sequences(seq, maxlen=self.max_len, padding="post", truncating="post")
        return padded

    def save(self, path="tokenizer.json"):
        """
        Save tokenizer to JSON file
        """
        with open(path, "w") as f:
            json.dump(self.tokenizer.to_json(), f)

        print(f"Tokenizer saved to {path}")

    def load(self, path="tokenizer.json"):
        """
        Load tokenizer from JSON file
        """
        from tensorflow.keras.preprocessing.text import tokenizer_from_json

        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer file not found at: {path}")

        with open(path) as f:
            data = json.load(f)

        self.tokenizer = tokenizer_from_json(data)
        self.word_index = self.tokenizer.word_index
        print(f"Tokenizer loaded from {path}")