from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def evaluate_model(y_true, y_pred):
    """
    Returns evaluation metrics
    """

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "confusion_matrix": cm
    }
