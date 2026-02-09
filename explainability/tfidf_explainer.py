import numpy as np

def tfidf_explain(model, vectorizer, processed_text, top_n=10):
    """
    Explain model prediction using TF-IDF feature importance
    """
    if not processed_text:
        return []

    feature_names = vectorizer.get_feature_names_out()
    vector = vectorizer.transform([processed_text]).toarray()[0]

    if not hasattr(model, "coef_"):
        return []

    weights = model.coef_[0]
    scores = vector * weights

    top_indices = np.argsort(scores)[-top_n:]

    explanation = [
        (feature_names[i], float(scores[i]))
        for i in reversed(top_indices)
        if scores[i] > 0
    ]

    return explanation
