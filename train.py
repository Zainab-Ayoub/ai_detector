import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from src.utils.tokenizer import TextTokenizer
from src.utils.helpers import load_data, save_tokenizer  # Changed load_dataset to load_data
from src.models.lstm_model import build_lstm_model
from src.models.neural_net import build_neural_net


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

MAX_VOCAB = 10000
MAX_LEN = 100
EMBED_DIM = 128
BATCH_SIZE = 32
EPOCHS = 5

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------
# TOKENIZATION + PREPROCESSING
# ---------------------------------------------------------

def prepare_data(texts, labels):
    """
    Tokenizes and converts text to padded sequences.
    """
    # Initialize tokenizer
    tokenizer = TextTokenizer(max_vocab=MAX_VOCAB)
    
    # Fit tokenizer on texts
    tokenizer.fit(texts)
    
    # Convert texts to sequences
    encoded = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences
    padded = pad_sequences(encoded, maxlen=MAX_LEN, padding="post")
    
    # Save tokenizer
    save_tokenizer(tokenizer.word_index)
    
    # Convert labels to one-hot
    num_classes = len(set(labels))
    y = to_categorical(labels, num_classes)
    
    return padded, y, num_classes


# ---------------------------------------------------------
# TRAINING
# ---------------------------------------------------------

def train():
    print("Loading dataset...")
    texts, labels = load_data()  # Changed from load_dataset to load_data
    
    print(f"Dataset loaded: {len(texts)} samples")
    
    print("Preparing data...")
    X, y, num_classes = prepare_data(texts, labels)
    
    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print(f"Number of classes: {num_classes}")
    
    # Train LSTM model
    print("\n" + "="*50)
    print("Training LSTM model...")
    print("="*50)
    lstm = build_lstm_model(MAX_VOCAB, EMBED_DIM, MAX_LEN, num_classes)
    lstm.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE,
        verbose=1
    )
    lstm.save(f"{MODEL_DIR}/lstm_model.h5")
    print(f"LSTM model saved to {MODEL_DIR}/lstm_model.h5")
    
    # Train Neural Net model
    print("\n" + "="*50)
    print("Training Neural Network model...")
    print("="*50)
    nn = build_neural_net(MAX_VOCAB, EMBED_DIM, MAX_LEN, num_classes)
    nn.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE,
        verbose=1
    )
    nn.save(f"{MODEL_DIR}/neural_net_model.h5")
    print(f"Neural Net model saved to {MODEL_DIR}/neural_net_model.h5")
    
    print("\n" + "="*50)
    print("✓ Training completed! All models saved in /models/ folder.")
    print("="*50)


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    train()