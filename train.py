import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ------------------------------
# LOAD DATA
# ------------------------------
data_path = "dataset/bias_dataset.csv"
df = pd.read_csv(data_path)

print("Dataset loaded:", df.shape)
print("Columns:", df.columns)

# ------------------------------
# CLEANING FUNCTION
# ------------------------------
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        return text
    return ""

# FIX COLUMN NAMES BASED ON YOUR DATASET
df["clean_text"] = df["Body"].apply(clean_text)
df["label_bias"] = df["Label"].apply(lambda x: 1 if x == 1 else 0)

print(df["label_bias"].value_counts())

# ------------------------------
# TRAIN/TEST SPLIT
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label_bias"], test_size=0.2, random_state=42
)

# ------------------------------
# TF-IDF
# ------------------------------
vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ------------------------------
# MODEL TRAINING
# ------------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# ------------------------------
# EVALUATION
# ------------------------------
preds = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# ------------------------------
# SAVE MODEL
# ------------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/bias_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("Model saved successfully.")
