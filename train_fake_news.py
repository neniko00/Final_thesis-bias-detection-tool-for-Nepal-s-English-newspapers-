import pandas as pd
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------
# CLEAN TEXT FUNCTION
# -------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# -------------------------
# 1. LOAD DATASET
# -------------------------
df = pd.read_csv("dataset/fake_news_clean.csv")

if "text" not in df.columns or "label" not in df.columns:
    raise ValueError("Dataset must contain 'text' and 'label' columns.")

df["clean_text"] = df["text"].apply(clean_text)

X = df["clean_text"]
y = df["label"]   # 0 = real, 1 = fake (based on your preprocessing)

# -------------------------
# 2. VECTORIZE TEXT
# -------------------------
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_vec = vectorizer.fit_transform(X)

# Save vectorizer
joblib.dump(vectorizer, "models/fake_tfidf_vectorizer.pkl")


# -------------------------
# 3. TRAIN MODELS
# -------------------------

models = {
    "logistic": LogisticRegression(max_iter=300),
    "svm": LinearSVC(),
    "naive_bayes": MultinomialNB(),
    "random_forest": RandomForestClassifier(n_estimators=200)
}

results = {}

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.25, random_state=42)

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"{name} Accuracy: {acc}")
    print(classification_report(y_test, preds))

    # Save model
    model_path = f"models/{name}_fake_model.pkl"
    joblib.dump(model, model_path)

    results[name] = acc

print("\nTraining completed successfully!")
print("Model Accuracies:")
print(results)
