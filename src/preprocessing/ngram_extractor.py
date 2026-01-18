import re
from collections import Counter
from src.utils.helpers import clean_text


class NgramExtractor:
    """
    Generates word-level and character-level n-grams.
    Useful for analyzing writing style differences between
    AI-generated and human-written text.
    """

    def __init__(self, max_features=3000):
        """
        max_features = number of n-grams to keep in vocabulary
        """
        self.max_features = max_features
        self.word_vocab = None
        self.char_vocab = None

    # ------------------------------------------------------
    # Basic n-gram extraction functions
    # ------------------------------------------------------

    def _extract_word_ngrams(self, text, n=2):
        words = clean_text(text).split()
        return [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]

    def _extract_char_ngrams(self, text, n=3):
        cleaned = clean_text(text)
        return [cleaned[i:i+n] for i in range(len(cleaned)-n+1)]

    # ------------------------------------------------------
    # Build vocabularies based on training data
    # ------------------------------------------------------

    def fit(self, texts):
        """
        Build vocabulary of most frequent n-grams across the dataset.
        """

        # Gather all n-grams
        word_grams = []
        char_grams = []

        for t in texts:
            word_grams.extend(self._extract_word_ngrams(t, n=2))
            word_grams.extend(self._extract_word_ngrams(t, n=3))
            char_grams.extend(self._extract_char_ngrams(t, n=3))
            char_grams.extend(self._extract_char_ngrams(t, n=4))

        # Select top-K most common n-grams
        self.word_vocab = [w for w, _ in Counter(word_grams).most_common(self.max_features)]
        self.char_vocab = [c for c, _ in Counter(char_grams).most_common(self.max_features)]

    # ------------------------------------------------------
    # Vectorize new text using the learned vocabulary
    # ------------------------------------------------------

    def transform(self, text):
        """
        Convert text into numeric feature vector:
        - word n-gram frequencies
        - char n-gram frequencies
        """

        if self.word_vocab is None or self.char_vocab is None:
            raise ValueError("NgramExtractor must be fitted before calling transform().")

        features = []

        # Extract n-grams
        word_grams = (
            self._extract_word_ngrams(text, 2)
            + self._extract_word_ngrams(text, 3)
        )
        char_grams = (
            self._extract_char_ngrams(text, 3)
            + self._extract_char_ngrams(text, 4)
        )

        word_counts = Counter(word_grams)
        char_counts = Counter(char_grams)

        # Build feature vector
        for w in self.word_vocab:
            features.append(word_counts[w])

        for c in self.char_vocab:
            features.append(char_counts[c])

        return features

    # ------------------------------------------------------
    # Batch transform (for datasets)
    # ------------------------------------------------------

    def transform_batch(self, texts):
        """
        Transform a list of texts into feature matrix.
        """
        return [self.transform(t) for t in texts]
