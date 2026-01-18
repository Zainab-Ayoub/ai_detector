import numpy as np
from tensorflow.keras.models import load_model


class EnsembleModel:
    """
    Loads multiple trained models and combines their predictions.
    Uses simple averaging (soft voting).
    """

    def __init__(self, model_paths):
        """
        model_paths: list of paths to trained model files (*.h5)
        """
        self.models = []
        for path in model_paths:
            print(f"Loading model: {path}")
            self.models.append(load_model(path))

        print(f"Total models loaded: {len(self.models)}")

    def predict(self, input_data):
        """
        Takes input_data (already preprocessed)
        and returns the averaged prediction of all models.
        """
        predictions = []

        # collect predictions from all models
        for model in self.models:
            pred = model.predict(input_data, verbose=0)
            predictions.append(pred)

        # convert to numpy array
        predictions = np.array(predictions)

        # average over axis 0 (model axis)
        avg_prediction = np.mean(predictions, axis=0)

        return avg_prediction

    def predict_class(self, input_data):
        """
        Returns the final class label after ensemble averaging.
        """
        avg_pred = self.predict(input_data)
        return np.argmax(avg_pred, axis=1)
