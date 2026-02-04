import numpy as np
from src.utils.helpers import clean_text, sentence_split
from src.preprocessing.ngram_extractor import NgramExtractor


class FeatureEngineer:
    """
    Combines multiple linguistic features + n-gram features
    to help classify AI vs Human writing.
    """

    def __init__(self, ngram_max_features=1500):
        self.ngram_extractor = NgramExtractor(max_features=ngram_max_features)
        self.fitted = False

    def fit(self, texts):
        """
        Fit n-gram vocabularies on training dataset.
        """
        self.ngram_extractor.fit(texts)
        self.fitted = True


    def basic_features(self, text):
        """
        Extract linguistic features representing writing style.
        """

        cleaned = clean_text(text)
        words = cleaned.split()
        sentences = sentence_split(text)

        word_count = len(words)
        sentence_count = len(sentences)

        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        avg_sentence_len = word_count / sentence_count if sentence_count > 0 else 0

        vocab_richness = len(set(words)) / word_count if word_count > 0 else 0

        punctuation_density = sum([1 for c in text if c in ".,!?"]) / max(len(text), 1)

        digit_ratio = sum([1 for c in text if c.isdigit()]) / max(len(text), 1)

        repeated_char_ratio = sum([1 for i in range(len(text)-1)
                                   if text[i] == text[i+1]]) / max(len(text), 1)

        return [
            word_count,
            sentence_count,
            avg_word_len,
            avg_sentence_len,
            vocab_richness,
            punctuation_density,
            digit_ratio,
            repeated_char_ratio,
        ]

    def transform(self, text):
        """
        Combine:
        - Linguistic features
        - N-gram features
        """
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted first!")

        base = self.basic_features(text)
        ngram = self.ngram_extractor.transform(text)

        return np.array(base + ngram, dtype=float)

    def transform_batch(self, texts):
        return np.array([self.transform(t) for t in texts])
