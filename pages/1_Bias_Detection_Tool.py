import streamlit as st
import joblib
import re
from textblob import TextBlob
from nrclex import NRCLex
import matplotlib.pyplot as plt

# ------------------------------
# Load Model + Vectorizer
# ------------------------------
MODEL_PATH = "models/bias_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# ------------------------------
# Load keyword & phrase list
# ------------------------------
def load_list(file):
    try:
        with open(file, "r") as f:
            return [w.strip().lower() for w in f.readlines()]
    except:
        return []

keywords = load_list("keywords.txt")
phrases = load_list("phrases.txt")

# ------------------------------
# Text Cleaning
# ------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ------------------------------
# Highlight Biased Words
# ------------------------------
def highlight_text(text, found_keywords, found_phrases):
    highlighted = text
    for word in found_keywords:
        highlighted = re.sub(
            rf"\b{word}\b",
            f"<span style='background-color:#ffcccc; padding:2px'>{word}</span>",
            highlighted,
            flags=re.IGNORECASE
        )
    for phrase in found_phrases:
        highlighted = re.sub(
            phrase,
            f"<span style='background-color:#ffcccc; padding:2px'>{phrase}</span>",
            highlighted,
            flags=re.IGNORECASE
        )
    return highlighted

# ------------------------------
# PAGE UI
# ------------------------------
st.title("🧠 Bias Detection Tool")
st.write("Analyze bias in any news article using ML + NLP + Emotion Analysis.")

text_input = st.text_area("Paste your article text here:")

if st.button("Analyze"):
    if not text_input.strip():
        st.error("Please enter some text.")
        st.stop()

    # CLEAN + VECTORIZE
    clean = clean_text(text_input)
    tfidf = vectorizer.transform([clean])

    # ML Prediction
    proba = model.predict_proba(tfidf)[0][1]
    ml_pred = model.predict(tfidf)[0]
    label = "Biased" if ml_pred == 1 else "Neutral"

    st.subheader("🔍 ML Prediction")
    st.write(f"**Prediction:** {label} ({proba:.2f} probability of bias)")

    # SENTIMENT ANALYSIS
    blob = TextBlob(text_input)
    sentiment = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    st.subheader("📘 Sentiment & Subjectivity")
    st.write(f"**Sentiment Score:** {sentiment:.2f}")
    st.write(f"**Subjectivity Score:** {subjectivity:.2f}")

    # EMOTION ANALYSIS
    emotion = NRCLex(text_input)
    emotion_scores = emotion.raw_emotion_scores

    st.subheader("🎭 Emotion Analysis")
    st.json(emotion_scores)

    # FIND KEYWORDS & PHRASES
    found_keywords = [w for w in keywords if w in clean]
    found_phrases = [p for p in phrases if p in clean]

    st.subheader("🔴 Biased Keywords & Phrases Found")
    if found_keywords or found_phrases:
        st.error(", ".join(found_keywords + found_phrases))
    else:
        st.success("No biased keywords detected.")

    # ------------------------------
    # BIAS SCORE (0-1)
    # ------------------------------
    ml_score = proba * 0.60
    subjectivity_score = subjectivity * 0.10
    sentiment_score = abs(sentiment) * 0.10

    emotion_boost = 0
    for emo in ["anger", "fear", "negative"]:
        if emo in emotion_scores:
            emotion_boost += 0.10

    keyword_score = 0.10 if (found_keywords or found_phrases) else 0

    bias_score = ml_score + subjectivity_score + sentiment_score + emotion_boost + keyword_score
    bias_score = min(bias_score, 1.0)

    st.subheader("🔥 Bias Score (0–1 Scale)")
    st.write(f"### {bias_score:.2f}")

    # FINAL CLASSIFICATION
    st.subheader("🏁 Final Classification")

    if bias_score >= 0.55:
        st.error("This article appears **BIASED**.")
    elif bias_score >= 0.30:
        st.warning("This article appears **SLIGHTLY BIASED**.")
    else:
        st.success("This article appears **NEUTRAL**.")

    # BIAS EXPLANATION
    st.subheader("🧠 Why was this decision made?")

    explanation = ""

    if found_keywords:
        explanation += f"- Biased keywords found: {', '.join(found_keywords)}.\n"

    if "anger" in emotion_scores or "fear" in emotion_scores or "negative" in emotion_scores:
        explanation += "- Strong negative emotions detected (anger/fear/negative).\n"

    explanation += f"- Bias Score ({bias_score:.2f}) indicates "

    if bias_score >= 0.55:
        explanation += "HIGH bias.\n"
    elif bias_score >= 0.30:
        explanation += "MODERATE bias.\n"
    else:
        explanation += "LOW/NO bias.\n"

    st.write(explanation)

    # ------------------------------
    # HIGHLIGHTED TEXT DISPLAY
    # ------------------------------
    st.subheader("📌 Highlighted Text (Biased Words)")

    highlighted = highlight_text(text_input, found_keywords, found_phrases)
    st.markdown(highlighted, unsafe_allow_html=True)

    # ------------------------------
    # VISUALIZATIONS
    # ------------------------------
    st.subheader("📊 Visualizations")

    # EMOTION BAR CHART
    if emotion_scores:
        emotions = list(emotion_scores.keys())
        values = list(emotion_scores.values())

        fig, ax = plt.subplots()
        ax.bar(emotions, values, color='skyblue')
        plt.xticks(rotation=45)
        plt.title("Emotion Analysis")
        st.pyplot(fig)

    # SENTIMENT METER
    st.write("### Sentiment Meter")
    fig2, ax2 = plt.subplots(figsize=(6, 1))
    ax2.barh(["Sentiment"], [sentiment], color="green" if sentiment > 0 else "red")
    ax2.set_xlim([-1, 1])
    st.pyplot(fig2)

