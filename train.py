import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from src.utils.tokenizer import TextTokenizer
from src.utils.helpers import load_data, save_tokenizer
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
    # Initialize tokenizer with correct parameter name
    tokenizer = TextTokenizer(max_vocab=MAX_VOCAB, max_len=MAX_LEN)
    
    # Fit tokenizer on texts
    print("Fitting tokenizer on texts...")
    tokenizer.fit(texts)
    
    # Convert texts to sequences
    print("Converting texts to sequences...")
    padded = tokenizer.texts_to_sequences(texts)
    
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
    print("="*60)
    print("STARTING TRAINING PROCESS")
    print("="*60)
    
    print("\n[1/5] Loading dataset...")
    texts, labels = load_data()
    
    print(f"\n✓ Dataset loaded: {len(texts)} samples")
    
    print("\n[2/5] Preparing data (tokenizing and padding)...")
    X, y, num_classes = prepare_data(texts, labels)
    
    print(f"\n✓ Data prepared:")
    print(f"  - Input shape: {X.shape}")
    print(f"  - Output shape: {y.shape}")
    print(f"  - Number of classes: {num_classes}")
    
    # Train/Val split
    print("\n[3/5] Splitting into train/validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Validation samples: {len(X_val)}")
    
    # Train LSTM model
    print("\n" + "="*60)
    print("[4/5] TRAINING LSTM MODEL")
    print("="*60)
    lstm = build_lstm_model(MAX_VOCAB, EMBED_DIM, MAX_LEN, num_classes)
    print("\nModel architecture:")
    lstm.summary()
    print("\nStarting training...")
    
    history_lstm = lstm.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE,
        verbose=1
    )
    lstm.save(f"{MODEL_DIR}/lstm_model.h5")
    print(f"\n✓ LSTM model saved to {MODEL_DIR}/lstm_model.h5")
    
    # Train Neural Net model
    print("\n" + "="*60)
    print("[5/5] TRAINING NEURAL NETWORK MODEL")
    print("="*60)
    nn = build_neural_net(MAX_VOCAB, EMBED_DIM, MAX_LEN, num_classes)
    print("\nModel architecture:")
    nn.summary()
    print("\nStarting training...")
    
    history_nn = nn.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE,
        verbose=1
    )
    nn.save(f"{MODEL_DIR}/neural_net_model.h5")
    print(f"\n✓ Neural Net model saved to {MODEL_DIR}/neural_net_model.h5")
    
    # Final summary
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nFinal Results:")
    print(f"  LSTM Model:")
    print(f"    - Training Accuracy: {history_lstm.history['accuracy'][-1]:.4f}")
    print(f"    - Validation Accuracy: {history_lstm.history['val_accuracy'][-1]:.4f}")
    print(f"  Neural Net Model:")
    print(f"    - Training Accuracy: {history_nn.history['accuracy'][-1]:.4f}")
    print(f"    - Validation Accuracy: {history_nn.history['val_accuracy'][-1]:.4f}")
    print(f"\nAll models saved in: {MODEL_DIR}/")
    print("="*60)


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    train()