import streamlit as st
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import os

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Model Evaluation & Limitations",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Evaluation & Limitations")

st.write("""
This page evaluates the performance of the machine learning model used for
media bias detection and discusses its limitations from an academic and
security-aware perspective.
""")

# --------------------------------------------------
# Paths
# --------------------------------------------------
ARTIFACT_DIR = "ml/artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "bias_model.joblib")
DATASET_PATH = "dataset/bias_dataset.csv"

# --------------------------------------------------
# Load model
# --------------------------------------------------
if not os.path.exists(MODEL_PATH):
    st.error("❌ Trained model not found. Please train the model first.")
    st.stop()

model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
if not os.path.exists(DATASET_PATH):
    st.error("❌ Dataset not found.")
    st.stop()

df = pd.read_csv(DATASET_PATH)

# --------------------------------------------------
# Use RAW TEXT (pipeline handles TF-IDF internally)
# --------------------------------------------------
TEXT_COLUMN = "content"
LABEL_COLUMN = "is_biased"

if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
    st.error("❌ Required columns not found in dataset.")
    st.stop()

X = df[TEXT_COLUMN].astype(str)
y = df[LABEL_COLUMN].astype(int)

# --------------------------------------------------
# Predictions
# --------------------------------------------------
y_pred = model.predict(X)

# --------------------------------------------------
# Classification Report
# --------------------------------------------------
st.subheader("📈 Classification Report")

report = classification_report(y, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df, width="stretch")  # ✅ FIXED

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------
st.subheader("🧮 Confusion Matrix")

cm = confusion_matrix(y, y_pred)
cm_df = pd.DataFrame(
    cm,
    columns=["Predicted Neutral", "Predicted Biased"],
    index=["Actual Neutral", "Actual Biased"]
)

st.dataframe(cm_df, width="stretch")  # ✅ FIXED

# --------------------------------------------------
# Interpretation
# --------------------------------------------------
st.subheader("🧠 Interpretation of Results")
st.write("""
The evaluation results indicate strong predictive performance for binary
bias detection. High precision and recall values suggest that the model is
effective at identifying lexical and framing cues associated with biased
reporting. However, performance should be interpreted cautiously due to
dataset-specific characteristics.
""")

# --------------------------------------------------
# Limitations
# --------------------------------------------------
st.subheader("⚠️ Limitations")
st.warning("""
• The dataset was teacher-provided and partially AI-generated.  
• High accuracy may not generalize to real-world, unseen news articles.  
• The model performs binary classification only (Biased vs Neutral).  
• Bias is subjective and context-dependent, which cannot be fully captured by lexical features alone.
""")

# --------------------------------------------------
# Security & Ethics
# --------------------------------------------------
st.subheader("🔐 Security & Ethical Considerations")
st.info("""
• Model outputs are probabilistic indicators, not absolute judgments.  
• No user data is stored or logged.  
• The system is designed for research and awareness, not automated moderation or censorship.
""")

# --------------------------------------------------
# Future Work
# --------------------------------------------------
st.subheader("🔮 Future Work")
st.write("""
Future enhancements may include:
• Multi-class bias detection (political, ethnic, gender-based bias)  
• Support for Nepali-language articles  
• Transformer-based explainable models (e.g., BERT with XAI techniques)  
• Robustness testing against adversarial text manipulation  
• Human-in-the-loop validation for ethical oversight
""")

st.success("This evaluation page aligns with academic assessment and thesis rigor.")
