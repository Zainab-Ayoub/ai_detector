import re
import nltk
from nltk.tokenize import word_tokenize

# Download tokenizer models on first use
nltk.download("punkt", quiet=True)


def clean_text(text):
    """
    Basic cleaning for raw text:
    - Convert to lowercase
    - Remove URLs
    - Remove special characters
    - Remove extra spaces

    This function is used before tokenization and model training.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)       # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", " ", text)  # Keep only simple characters
    text = re.sub(r"\s+", " ", text).strip()         # Remove extra spaces

    return text


def tokenize(text):
    """
    Word-level tokenization (ONLY for debugging/analysis).
    Not used inside the main model pipeline.
    """
    cleaned = clean_text(text)
    return word_tokenize(cleaned)


def sentence_split(text):
    """
    Split text into sentences.
    Useful for research/analysis, but not required for the model.
    """
    cleaned = clean_text(text)
    sentences = re.split(r"[.!?]", cleaned)
    return [s.strip() for s in sentences if s.strip()]
