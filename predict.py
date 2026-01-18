import pickle
from src.utils.helpers import clean_text

# Load the trained model
with open("models/logistic_model_full.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer_full.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def predict_text(text):
    """
    Predict if text is AI-generated or Human-written
    Returns: 'AI' or 'Human' with confidence score
    """
    # Clean the text
    cleaned = clean_text(text)
    
    # Vectorize
    features = vectorizer.transform([cleaned])
    
    # Predict
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    label = "AI" if prediction == 1 else "Human"
    confidence = probabilities[prediction] * 100
    
    return label, confidence

# Example usage
if __name__ == "__main__":
    # Test with sample text
    sample_text = """
    The quick brown fox jumps over the lazy dog. This is a test sentence 
    to demonstrate the AI detection capabilities of our model.
    """
    
    label, confidence = predict_text(sample_text)
    print(f"\nPrediction: {label}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Interactive mode
    print("\n" + "="*60)
    print("AI TEXT DETECTOR - Interactive Mode")
    print("="*60)
    print("Enter text to check (or 'quit' to exit):\n")
    
    while True:
        text = input("\nEnter text: ")
        if text.lower() == 'quit':
            break
        
        if len(text.strip()) < 10:
            print("Please enter at least 10 characters")
            continue
        
        label, confidence = predict_text(text)
        print(f"\n✓ Prediction: {label}")
        print(f"✓ Confidence: {confidence:.2f}%")