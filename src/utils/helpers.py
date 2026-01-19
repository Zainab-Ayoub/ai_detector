import os
import re
import pandas as pd

# Correct dataset path
DATASET_PATH = "data/master_training_data.csv"


def sentence_split(text):
    """
    Split text into sentences using punctuation markers.
    This is needed by feature_engineering.py
    """
    sentences = re.split(r'(?<=[.!?])\s+', str(text).strip())
    return [s.strip() for s in sentences if s.strip()]


def clean_text(text):
    """
    Basic text cleaning function.
    """
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def load_data():
    """
    Loads the cleaned + balanced dataset created earlier.
    Returns (texts, labels)
    """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    
    # Clean column names (remove extra spaces and fix malformed headers)
    df.columns = df.columns.str.strip()
    
    # Fix the malformed 'text' column name
    if '5 877=text' in df.columns:
        df.rename(columns={'5 877=text': 'text'}, inplace=True)
    
    # Also handle any column that ends with '=text'
    for col in df.columns:
        if col.endswith('=text'):
            df.rename(columns={col: 'text'}, inplace=True)
            break
    
    print(f"Available columns: {df.columns.tolist()}")
    print(f"Dataset shape: {df.shape}")

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Dataset must contain 'text' and 'label' columns. Found: {df.columns.tolist()}")

    # Remove any rows with missing text or label
    df = df.dropna(subset=["text", "label"])
    
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    
    print(f"Loaded {len(texts)} samples with {len(set(labels))} unique labels")

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