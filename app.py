import streamlit as st
import pandas as pd
import joblib

# -------------------- CUSTOM MODULE IMPORTS --------------------
from nlp.preprocessing import preprocess_text
from security.input_validation import validate_input
from evaluation.metrics import evaluate_model
# --------------------------------------------------------------

# -------------------- PAGE CONFIG ------------------------------
st.set_page_config(
    page_title="Bias Detection System",
    page_icon="📰",
    layout="wide"
)
# --------------------------------------------------------------

# -------------------- LOAD MODEL -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("ml/artifacts/bias_model.joblib")
    return model

model = load_model()
# --------------------------------------------------------------

# -------------------- SIDEBAR NAVIGATION -----------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Module",
    ["Home", "Bias Detection", "Model Evaluation"]
)
# --------------------------------------------------------------

# ============================ HOME =============================
if page == "Home":
    st.title("📰 Bias Detection Dashboard")
    st.write("Welcome to the Bias Detection System for Nepali & International News.")

    st.markdown("""
    ### 📌 What this system includes:
    - **Bias Detection Tool**
    - **Text Preprocessing (Tokenization & Lemmatization)**
    - **Machine Learning Bias Classification**
    - **Explainable AI (Feature Importance)**
    - **Model Evaluation (Accuracy, F1, Confusion Matrix)**
    - **Secure Input Handling**
    - **Streamlit-Based Interactive Dashboard**
    """)

    st.info("➡️ Select a module from the sidebar to begin.")

# ======================= BIAS DETECTION ========================
elif page == "Bias Detection":
    st.title("🧠 Bias Detection Tool")
    st.write("Enter an English news article to analyze bias.")

    user_input = st.text_area(
        "📝 Paste news article text here:",
        height=250
    )

    if st.button("Analyze Bias"):
        try:
            # -------- SECURITY VALIDATION --------
            safe_text = validate_input(user_input)

            # -------- NLP PREPROCESSING ----------
            processed_text = preprocess_text(safe_text)

            # -------- PREDICTION (PIPELINE) ------
            prediction = model.predict([processed_text])[0]
            probability = model.predict_proba([processed_text])[0].max()

            st.subheader("🔍 Prediction Result")

            if prediction == 1:
                st.error(f"⚠️ Biased Article (Confidence: {probability:.2f})")
            else:
                st.success(f"✅ Neutral Article (Confidence: {probability:.2f})")

            # -------- EXPLAINABILITY -------------
            st.subheader("📊 Feature Importance (Top Influential Words)")

            if hasattr(model, "named_steps"):
                vectorizer = model.named_steps.get("tfidf", None)
                classifier = model.named_steps.get("clf", None)

                if vectorizer and classifier is not None:
                    feature_names = vectorizer.get_feature_names_out()
                    coefficients = classifier.coef_[0]

                    feature_df = pd.DataFrame({
                        "Word": feature_names,
                        "Importance": coefficients
                    })

                    top_features = feature_df.reindex(
                        feature_df.Importance.abs().sort_values(ascending=False).index
                    ).head(10)

                    st.dataframe(top_features)
                else:
                    st.info("Explainability not available for this model.")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ===================== MODEL EVALUATION ========================
elif page == "Model Evaluation":
    st.title("📈 Model Evaluation")
    st.write("This section evaluates the trained bias classification model.")

    # -------- LOAD DATASET --------
    data = pd.read_csv("dataset/bias_dataset.csv")

    # -------- AUTO-DETECT TEXT COLUMN --------
    possible_text_cols = ["text", "content", "article", "clean_text", "news", "full_text"]
    text_column = next((col for col in possible_text_cols if col in data.columns), None)

    if text_column is None:
        st.error("❌ No valid text column found in dataset.")
        st.stop()

    # -------- AUTO-DETECT LABEL COLUMN --------
    possible_label_cols = ["label", "is_biased", "bias_label"]
    label_column = next((col for col in possible_label_cols if col in data.columns), None)

    if label_column is None:
        st.error("❌ No valid label column found in dataset.")
        st.stop()

    X = data[text_column]
    y_true = data[label_column]

    # -------- PREPROCESS TEXT --------
    X_processed = X.apply(preprocess_text)

    # -------- PREDICT (PIPELINE) ----
    y_pred = model.predict(X_processed)

    # -------- EVALUATE --------------
    metrics = evaluate_model(y_true, y_pred)

    st.subheader("📊 Evaluation Metrics")
    st.metric("Accuracy", round(metrics["accuracy"], 2))
    st.metric("F1 Score", round(metrics["f1_score"], 2))

    st.subheader("🧮 Confusion Matrix")
    st.write(metrics["confusion_matrix"])
