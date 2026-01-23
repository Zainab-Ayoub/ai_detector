import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

from helpers import load_dataset, load_tokenizer
from tokenizers import tokenize

MAX_LEN = 100
MODEL_DIR = "models"


def encode_texts(texts, word_index):
    encoded = []
    for t in texts:
        tokens = tokenize(t)
        encoded.append([word_index.get(w, 0) for w in tokens])
    return pad_sequences(encoded, maxlen=MAX_LEN, padding="post")


def ensemble_predict(preds):
    """
    preds = list of (num_samples x num_classes) arrays
    Uses majority vote across models.
    """
    pred_classes = [np.argmax(p, axis=1) for p in preds]
    stacked = np.stack(pred_classes, axis=1)

    # Majority vote along each row
    final = []
    for row in stacked:
        values, counts = np.unique(row, return_counts=True)
        final.append(values[np.argmax(counts)])
    return np.array(final)


def print_metrics(y_true, y_pred, name):
    print(f"\n🔹 {name} Metrics")
    print("--------------------------------")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average="weighted"))
    print("Recall   :", recall_score(y_true, y_pred, average="weighted"))
    print("F1 Score :", f1_score(y_true, y_pred, average="weighted"))


def evaluate():
    print("Loading dataset...")
    texts, labels = load_dataset()

    # Load tokenizer
    print("Loading tokenizer...")
    word_index = load_tokenizer()

    # Encode text
    print("Encoding text...")
    X = encode_texts(texts, word_index)
    y_true = np.array(labels)

    print("Loading models...")
    lstm = load_model(f"{MODEL_DIR}/lstm_model.h5")
    gru = load_model(f"{MODEL_DIR}/gru_model.h5")
    cnn = load_model(f"{MODEL_DIR}/cnn_model.h5")

 
    print("Running predictions...")

    lstm_pred = lstm.predict(X, verbose=0)
    gru_pred  = gru.predict(X, verbose=0)
    cnn_pred  = cnn.predict(X, verbose=0)

    lstm_labels = np.argmax(lstm_pred, axis=1)
    gru_labels  = np.argmax(gru_pred, axis=1)
    cnn_labels  = np.argmax(cnn_pred, axis=1)

    print_metrics(y_true, lstm_labels, "LSTM")
    print_metrics(y_true, gru_labels, "GRU")
    print_metrics(y_true, cnn_labels, "CNN")

    print("\n🔮 Running Ensemble (Majority Vote)...")
    final_pred = ensemble_predict([lstm_pred, gru_pred, cnn_pred])

    print_metrics(y_true, final_pred, "Ensemble")

    print("\n🎉 Evaluation completed!")


if __name__ == "__main__":
    evaluate()
