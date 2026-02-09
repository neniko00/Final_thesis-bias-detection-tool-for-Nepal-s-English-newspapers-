import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------------------------
# Paths
# --------------------------------------------------
DATASET_PATH = "dataset/bias_dataset.csv"
MODEL_DIR = "ml/artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "bias_model.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
df = pd.read_csv(DATASET_PATH)

print("📌 Dataset loaded successfully")
print("📌 Total samples:", len(df))

# --------------------------------------------------
# FIXED COLUMN MAPPING (based on your dataset)
# --------------------------------------------------
TEXT_COLUMN = "content"
LABEL_COLUMN = "is_biased"

# Safety check
if TEXT_COLUMN not in df.columns:
    raise ValueError(f"❌ Text column '{TEXT_COLUMN}' not found")

if LABEL_COLUMN not in df.columns:
    raise ValueError(f"❌ Label column '{LABEL_COLUMN}' not found")

X = df[TEXT_COLUMN].astype(str)
y = df[LABEL_COLUMN].astype(int)

print(f"✅ Using TEXT column: {TEXT_COLUMN}")
print(f"✅ Using LABEL column: {LABEL_COLUMN}")

# --------------------------------------------------
# Train-test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# ML Pipeline (TF-IDF + Logistic Regression)
# --------------------------------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_df=0.9,
        min_df=3,
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear"
    ))
])

# --------------------------------------------------
# Train model
# --------------------------------------------------
print("\n🚀 Training model...")
pipeline.fit(X_train, y_train)

# --------------------------------------------------
# Evaluate model
# --------------------------------------------------
y_pred = pipeline.predict(X_test)

print("\n📊 Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n📄 Classification Report:\n")
print(classification_report(y_test, y_pred))

# --------------------------------------------------
# Save model
# --------------------------------------------------
# Save full pipeline
joblib.dump(pipeline, MODEL_PATH)

# Save TF-IDF vectorizer separately (for Streamlit pages)
VEC_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
joblib.dump(pipeline.named_steps["tfidf"], VEC_PATH)

print(f"\n✅ Model saved at: {MODEL_PATH}")
print(f"✅ Vectorizer saved at: {VEC_PATH}")

print(f"\n✅ Model saved successfully at:\n{MODEL_PATH}")
