"""
SHAP explainability module (FINAL – stable & Streamlit-safe)
Works with TF-IDF + linear models (Logistic Regression / Linear SVM)
"""

import shap
import numpy as np


def shap_explain(model, vectorizer, processed_text, top_n=15):
    """
    Generate SHAP feature-importance explanation.

    Returns:
        top_features: list of (feature, shap_value)
        shap_values: SHAP Explanation object
        explainer: SHAP explainer instance
    """

    if not processed_text:
        return None, None, None

    # Vectorize input
    X = vectorizer.transform([processed_text])

    # Modern SHAP masker (removes FutureWarning)
    masker = shap.maskers.Independent(X)

    # Linear explainer (correct for TF-IDF models)
    explainer = shap.LinearExplainer(model, masker)

    shap_values = explainer(X)

    feature_names = vectorizer.get_feature_names_out()
    values = shap_values.values[0]

    # Top-N influential features
    top_idx = np.argsort(np.abs(values))[-top_n:]

    top_features = [
        (feature_names[i], float(values[i]))
        for i in reversed(top_idx)
    ]

    return top_features, shap_values, explainer
