import streamlit as st
from fpdf import FPDF
from textblob import TextBlob
from nrclex import NRCLex
import joblib
import re
import tempfile
import os

# ----------------------------------------------------
# STREAMLIT PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="Download Bias Report",
    page_icon="📄",
    layout="centered"
)

# ----------------------------------------------------
# LOAD MODEL + TF-IDF VECTORIZER
# ----------------------------------------------------
model = joblib.load("models/bias_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# ----------------------------------------------------
# CLEAN TEXT FUNCTION
# ----------------------------------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ----------------------------------------------------
# PDF CREATOR FUNCTION
# ----------------------------------------------------
def create_pdf(article_text, bias_prob, sentiment, subjectivity, emotions):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Bias Detection Report", ln=True)
    pdf.ln(5)

    # Article Text
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, f"Original Article:\n{article_text}")
    pdf.ln(5)

    # Bias Probability
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Bias Probability", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, f"{bias_prob:.4f}")
    pdf.ln(3)

    # Sentiment
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Sentiment Score", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, f"{sentiment:.4f}")
    pdf.ln(3)

    # Subjectivity
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Subjectivity Score", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, f"{subjectivity:.4f}")
    pdf.ln(3)

    # Emotions
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Emotion Scores", ln=True)
    pdf.set_font("Arial", size=12)

    if emotions:
        for emo, val in emotions.items():
            pdf.cell(0, 8, f"{emo}: {val}", ln=True)
    else:
        pdf.cell(0, 8, "No dominant emotions detected.", ln=True)

    # Save PDF to temp file
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_pdf.name)

    return temp_pdf.name

# ----------------------------------------------------
# STREAMLIT UI
# ----------------------------------------------------
st.title("📄 Download PDF Report")
st.write("Generate an automatic bias analysis report for any English news article.")

article_text = st.text_area(
    "Paste article text here:",
    height=250
)

if st.button("Generate Report"):
    if not article_text.strip():
        st.error("Please paste an article first.")
        st.stop()

    # Preprocess
    cleaned_text = clean_text(article_text)
    vector = vectorizer.transform([cleaned_text])

    # Bias Probability
    bias_prob = model.predict_proba(vector)[0][1]

    # Sentiment & Subjectivity
    blob = TextBlob(article_text)
    sentiment = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    # Emotion Analysis
    emotions = NRCLex(article_text).raw_emotion_scores

    # Generate PDF
    pdf_path = create_pdf(
        article_text,
        bias_prob,
        sentiment,
        subjectivity,
        emotions
    )

    # Download Button
    with open(pdf_path, "rb") as file:
        st.download_button(
            label="📄 Download PDF Report",
            data=file,
            file_name="bias_report.pdf",
            mime="application/pdf"
        )

    st.success("PDF report generated successfully!")

    # Optional cleanup
    os.remove(pdf_path)
