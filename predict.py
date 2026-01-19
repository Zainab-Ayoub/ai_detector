import os
import pickle
from src.utils.helpers import clean_text

# Model artifact locations
MODEL_PATH = os.path.join("models", "logistic_model_full.pkl")
VECTORIZER_PATH = os.path.join("models", "vectorizer_full.pkl")

_model = None
_vectorizer = None


def _load_artifacts():
    """
    Lazy-load model artifacts to avoid import-time failures.
    """
    global _model, _vectorizer
    if _model is not None and _vectorizer is not None:
        return _model, _vectorizer

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Train the model first."
        )
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            f"Vectorizer file not found: {VECTORIZER_PATH}. Train the model first."
        )

    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        _vectorizer = pickle.load(f)

    return _model, _vectorizer


def _append_warning(base, message):
    if not message:
        return base
    if not base:
        return message
    return f"{base}\n{message}"


# Confidence thresholds
HIGH_CONFIDENCE_THRESHOLD = 70  # Above this = confident prediction
LOW_CONFIDENCE_THRESHOLD = 60   # Below this = uncertain, needs review
MIN_WORDS_FOR_ACCURACY = 50     # Minimum words for reliable prediction


def predict_text(text):
    """
    Predict if text is AI-generated or Human-written
    Returns: prediction label, confidence score, and warning message
    """
    model, vectorizer = _load_artifacts()

    # Clean the text
    cleaned = clean_text(text)
    word_count = len(cleaned.split())
    
    # Vectorize
    features = vectorizer.transform([cleaned])
    
    # Predict
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # Get confidence (probability of the predicted class)
    confidence = probabilities[prediction] * 100
    
    # Determine label and warning
    warning = None
    
    # Check for short text
    if word_count < MIN_WORDS_FOR_ACCURACY:
        warning = (
            f"WARNING: Text is short ({word_count} words). "
            "Results may be unreliable. For best accuracy, use 50+ words."
        )
    
    # Check confidence level
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        label = "UNCERTAIN"
        warning = _append_warning(
            warning, "Model confidence is too low. This text needs human review."
        )
    elif confidence < HIGH_CONFIDENCE_THRESHOLD:
        label = "AI" if prediction == 1 else "Human"
        warning = _append_warning(
            warning, "Moderate confidence - consider reviewing manually."
        )
    else:
        label = "AI" if prediction == 1 else "Human"
    
    return label, confidence, probabilities, warning, word_count


def get_detailed_prediction(text):
    """
    Get a detailed prediction with all relevant information.
    """
    label, confidence, probs, warning, word_count = predict_text(text)
    
    result = {
        'prediction': label,
        'confidence': confidence,
        'human_probability': probs[0] * 100,
        'ai_probability': probs[1] * 100,
        'word_count': word_count,
        'warning': warning,
        'needs_review': confidence < HIGH_CONFIDENCE_THRESHOLD
    }
    
    return result


def read_multiline_input():
    """
    Read multi-line input from the user.
    End input with a line containing only END.
    """
    print("Paste your text. End by typing END on its own line.")
    lines = []
    while True:
        line = input()
        if line.strip().lower() == "quit":
            return None
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


# Example usage
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AI TEXT DETECTOR")
    print("  Detects if text was written by Human or AI")
    print("="*60)
    
    print("\nNOTE: For best accuracy, provide at least 50 words.")
    print("      Short texts (like song lyrics) may be unreliable.")
    print("\nEnter text to check (or 'quit' to exit):\n")
    
    while True:
        print("-"*60)
        print("\nEnter text:")
        text = read_multiline_input()
        
        if text is None:
            print("\nGoodbye!")
            break
        
        if not text or len(text.strip()) < 10:
            print("Please enter at least 10 characters")
            continue
        
        # Get prediction
        result = get_detailed_prediction(text)
        
        # Display results
        print("\n" + "="*60)
        print("  RESULTS")
        print("="*60)
        
        print(f"\n  Prediction:    {result['prediction']}")
        print(f"  Confidence:    {result['confidence']:.1f}%")
        print(f"  Word Count:    {result['word_count']}")
        print(f"\n  Probabilities:")
        print(f"    - Human: {result['human_probability']:.1f}%")
        print(f"    - AI:    {result['ai_probability']:.1f}%")
        
        if result['warning']:
            print(f"\n  {result['warning']}")
        
        if result['needs_review']:
            print("\n  [!] RECOMMENDATION: This result should be reviewed by a human.")
        
        print("="*60)
