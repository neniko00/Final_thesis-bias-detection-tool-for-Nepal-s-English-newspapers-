"""
Explainability factory module
Provides a unified interface for different explanation methods
"""

from explainability.tfidf_explainer import tfidf_explain
from explainability.shap_explainer import shap_explain


def generate_explanation(model, vectorizer, processed_text, method="tfidf"):
    """
    Generate explanation based on selected explainability method

    Parameters:
        model: trained ML model
        vectorizer: TF-IDF vectorizer
        processed_text (str): preprocessed input text
        method (str): 'tfidf' or 'shap'

    Returns:
        TF-IDF → list of (word, score)
        SHAP  → (shap_values, explainer)
    """

    if not processed_text:
        return None

    method = method.lower()

    if method == "shap":
        return shap_explain(model, vectorizer, processed_text)

    # Default: TF-IDF explanation
    return tfidf_explain(model, vectorizer, processed_text)
