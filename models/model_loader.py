"""
Model loader utility
Loads trained bias detection model and TF-IDF vectorizer
"""

import joblib
import os


def load_bias_model():
    """
    Load the trained bias detection model and TF-IDF vectorizer.

    Returns:
        model, vectorizer
    """

    model_path = os.path.join("models", "bias_model.pkl")
    vectorizer_path = os.path.join("models", "tfidf_vectorizer.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer
