import os
import pandas as pd

# Correct dataset path
DATASET_PATH = "data/master_training_data.csv"


def load_data():
    """
    Loads the cleaned + balanced dataset created earlier.
    Returns (texts, labels)
    """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns")

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    return texts, labels


def split_dataset(texts, labels, test_size=0.2, seed=42):
    """
    Splits the dataset manually.
    """
    import numpy as np

    np.random.seed(seed)
    indices = np.random.permutation(len(texts))
    split_point = int(len(texts) * (1 - test_size))

    train_idx = indices[:split_point]
    test_idx = indices[split_point:]

    train_texts = [texts[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    train_labels = [labels[i] for i in train_idx]
    test_labels = [labels[i] for i in test_idx]

    return train_texts, test_texts, train_labels, test_labels


def save_tokenizer(word_index, path="outputs/tokenizer.json"):
    """
    Saves the tokenizer dictionary.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    import json
    with open(path, "w") as f:
        json.dump(word_index, f)

    print(f"Tokenizer saved to {path}")
