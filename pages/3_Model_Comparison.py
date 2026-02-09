import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

st.title("📊 Model Comparison Dashboard")
st.write("Performance comparison of machine learning models used for bias detection.")

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
DATASET_PATH = "dataset/bias_dataset.csv"

df = pd.read_csv(DATASET_PATH)

# Use correct columns from your dataset
X = df["content"].astype(str)
y = df["is_biased"].astype(int)

# --------------------------------------------------
# Train-test split (RAW TEXT – pipeline handles TF-IDF)
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------------
# Load Available Models
# --------------------------------------------------
models = {}

PIPELINE_PATH = "ml/artifacts/bias_model.joblib"

if os.path.exists(PIPELINE_PATH):
    models["Logistic Regression (TF-IDF)"] = joblib.load(PIPELINE_PATH)

if not models:
    st.error("❌ No trained models found. Please train the model first.")
    st.stop()

# --------------------------------------------------
# Evaluate Models
# --------------------------------------------------
results = {}

for model_name, model in models.items():
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    results[model_name] = {
        "Accuracy": acc,
        "Precision": report["1"]["precision"],
        "Recall": report["1"]["recall"],
        "F1 Score": report["1"]["f1-score"],
    }

df_results = pd.DataFrame(results).T

# --------------------------------------------------
# Display Results
# --------------------------------------------------
st.subheader("📌 Model Performance Table")
st.dataframe(
    df_results.style.format("{:.4f}"),
    width="stretch"   # ✅ FIXED (no deprecation warning)
)

# --------------------------------------------------
# Accuracy Chart
# --------------------------------------------------
st.subheader("📈 Accuracy Comparison")

fig, ax = plt.subplots()
ax.bar(
    df_results.index,
    df_results["Accuracy"],
    color="#4CAF50"
)
plt.xticks(rotation=30)
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")

st.pyplot(fig, width="stretch")  # ✅ FIXED

# --------------------------------------------------
# Best Model Highlight
# --------------------------------------------------
best_model = df_results["Accuracy"].idxmax()
best_acc = df_results["Accuracy"].max()

st.success(f"🏆 **Best Model:** {best_model}  |  **Accuracy:** {best_acc:.4f}")
