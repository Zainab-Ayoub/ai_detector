import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

from src.utils.helpers import load_data, clean_text


# ---------------------------------------------------------
# CONFIGURATION - FULL TRAINING
# ---------------------------------------------------------

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------
# TRAINING WITH FULL DATASET
# ---------------------------------------------------------

def train():
    print("="*60)
    print("FULL TRAINING MODE - ALL DATA")
    print("="*60)
    
    print("\n[1/4] Loading dataset...")
    texts, labels = load_data()
    
    print(f"✓ Dataset loaded: {len(texts)} samples")
    
    print("\n[2/4] Cleaning texts...")
    texts_clean = [clean_text(t) for t in texts]
    print("✓ Texts cleaned")
    
    print("\n[3/4] Converting to TF-IDF features...")
    print("  This may take 1-2 minutes...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
    X = vectorizer.fit_transform(texts_clean)
    print(f"✓ Features created: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )
    
    print(f"  - Training samples: {len(y_train)}")
    print(f"  - Test samples: {len(y_test)}")
    
    print("\n[4/4] Training Logistic Regression model...")
    print("  This may take 2-3 minutes...")
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("✓ Model trained!")
    
    # Evaluate
    print("\nEvaluating model...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nResults:")
    print(f"  - Total samples: {len(texts)}")
    print(f"  - Training samples: {len(y_train)}")
    print(f"  - Test samples: {len(y_test)}")
    print(f"  - Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"  - Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    print("\nDetailed Test Set Results:")
    print(classification_report(y_test, y_pred_test, target_names=['Human', 'AI']))
    
    # Save model
    print(f"\nSaving model to {MODEL_DIR}/...")
    with open(f"{MODEL_DIR}/logistic_model_full.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{MODEL_DIR}/vectorizer_full.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    print(f"✓ Model saved to {MODEL_DIR}/logistic_model_full.pkl")
    print(f"✓ Vectorizer saved to {MODEL_DIR}/vectorizer_full.pkl")
    print("\n" + "="*60)
    print("PRODUCTION MODEL READY!")
    print("="*60)


if __name__ == "__main__":
    train()