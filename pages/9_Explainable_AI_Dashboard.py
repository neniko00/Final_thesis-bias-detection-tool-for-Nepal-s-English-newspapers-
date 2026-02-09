import streamlit as st

from config.settings import NLP_ENGINE, EXPLAINABILITY_ENGINE
from nlp.nlp_factory import preprocess_text_factory
from explainability.explainability_factory import generate_explanation
from models.model_loader import load_bias_model

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Explainable AI Dashboard",
    layout="wide"
)

st.title("🧠 Explainable AI – Bias Detection Dashboard")
st.write(
    "This page explains *why* a news article is classified as biased or neutral "
    "using interpretable machine learning techniques."
)

# --------------------------------------------------
# Safe model loading (prevents blank page)
# --------------------------------------------------
try:
    model, vectorizer = load_bias_model()
except Exception as e:
    st.error("❌ Failed to load model or vectorizer.")
    st.code(str(e))
    st.stop()

# --------------------------------------------------
# Input
# --------------------------------------------------
input_text = st.text_area(
    "Enter an English news article",
    height=250
)

# --------------------------------------------------
# Action
# --------------------------------------------------
if st.button("Analyze and Explain"):
    if not input_text.strip():
        st.warning("Please enter article text.")
        st.stop()

    processed_text = preprocess_text_factory(
        input_text, NLP_ENGINE
    )

    explanation = generate_explanation(
        model,
        vectorizer,
        processed_text,
        EXPLAINABILITY_ENGINE
    )

    # --------------------------------------------------
    # SHAP Explanation (FINAL & SAFE)
    # --------------------------------------------------
    if EXPLAINABILITY_ENGINE == "shap":
        top_features, shap_values, explainer = explanation

        if top_features is None:
            st.error("SHAP explanation could not be generated.")
        else:
            st.subheader("🔍 SHAP Feature Importance")

            st.write(
                "The following words contributed most to the model’s prediction. "
                "Positive values indicate stronger influence toward the predicted class."
            )

            for word, value in top_features:
                st.write(f"**{word}** → SHAP value: {value:.4f}")

    # --------------------------------------------------
    # TF-IDF Explanation (Fallback / Alternative)
    # --------------------------------------------------
    else:
        st.subheader("🔍 TF-IDF Feature Importance")

        if explanation:
            for word, score in explanation:
                st.write(f"**{word}** → Score: {score:.4f}")
        else:
            st.info("No significant features found.")

st.success("Explainability analysis completed successfully.")
