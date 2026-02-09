import streamlit as st
import matplotlib.pyplot as plt
from textblob import TextBlob
from nrclex import NRCLex
from wordcloud import WordCloud
import joblib
import re

# ------------------------------
# Load Model & Vectorizer
# ------------------------------
model = joblib.load("models/bias_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# ------------------------------
# Cleaning Function
# ------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


st.title("📰 Newspaper Comparison Dashboard")
st.write("Compare bias, sentiment, subjectivity, and emotions between two news sources.")

# ------------------------------
# Input Fields
# ------------------------------
col1, col2 = st.columns(2)

with col1:
    text1 = st.text_area("📰 Article 1 (Source A)", height=220)

with col2:
    text2 = st.text_area("📰 Article 2 (Source B)", height=220)

# ------------------------------
# Process Button
# ------------------------------
if st.button("Compare Articles"):

    if not text1.strip() or not text2.strip():
        st.error("Please paste text in BOTH articles.")
    else:
        # ------------------------------
        # Clean & Predict
        # ------------------------------
        clean1 = clean_text(text1)
        clean2 = clean_text(text2)

        vec1 = vectorizer.transform([clean1])
        vec2 = vectorizer.transform([clean2])

        bias1 = model.predict_proba(vec1)[0][1]
        bias2 = model.predict_proba(vec2)[0][1]

        # ------------------------------
        # Sentiment & Subjectivity
        # ------------------------------
        sent1 = TextBlob(text1).sentiment
        sent2 = TextBlob(text2).sentiment

        # ------------------------------
        # Emotions
        # ------------------------------
        emo1 = NRCLex(text1).raw_emotion_scores
        emo2 = NRCLex(text2).raw_emotion_scores

        # ------------------------------
        # Bias Comparison
        # ------------------------------
        st.subheader("📊 Bias Score Comparison")
        st.write(f"**Source A Bias Score:** {bias1:.2f}")
        st.write(f"**Source B Bias Score:** {bias2:.2f}")

        fig, ax = plt.subplots()
        ax.bar(["Source A", "Source B"], [bias1, bias2], color=["red", "blue"])
        ax.set_title("Bias Comparison")
        ax.set_ylabel("Bias Score (0–1)")
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        # ------------------------------
        # Sentiment Comparison
        # ------------------------------
        st.subheader("🙂 Sentiment Comparison")

        fig2, ax2 = plt.subplots()
        ax2.bar(
            ["Source A", "Source B"],
            [sent1.polarity, sent2.polarity],
            color=["green", "purple"]
        )
        ax2.set_title("Sentiment Polarity (-1 to +1)")
        st.pyplot(fig2, width="stretch")
        plt.close(fig2)

        # ------------------------------
        # Subjectivity Comparison
        # ------------------------------
        st.subheader("🧠 Subjectivity Comparison")

        fig3, ax3 = plt.subplots()
        ax3.bar(
            ["Source A", "Source B"],
            [sent1.subjectivity, sent2.subjectivity],
            color=["orange", "cyan"]
        )
        ax3.set_title("Subjectivity (0 = Objective, 1 = Subjective)")
        st.pyplot(fig3, width="stretch")
        plt.close(fig3)

        # ------------------------------
        # Emotion Comparison
        # ------------------------------
        st.subheader("❤️ Emotion Comparison (Top Emotions)")

        all_emotions = sorted(set(emo1.keys()) | set(emo2.keys()))
        data1 = [emo1.get(e, 0) for e in all_emotions]
        data2 = [emo2.get(e, 0) for e in all_emotions]

        fig4, ax4 = plt.subplots(figsize=(10, 4))
        ax4.plot(all_emotions, data1, label="Source A", marker="o")
        ax4.plot(all_emotions, data2, label="Source B", marker="o")
        ax4.set_title("Emotion Intensity")
        ax4.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig4, width="stretch")
        plt.close(fig4)

        # ------------------------------
        # WordCloud Comparison
        # ------------------------------
        st.subheader("☁ WordCloud Comparison")

        wc1 = WordCloud(background_color="white").generate(clean1)
        wc2 = WordCloud(background_color="white").generate(clean2)

        colA, colB = st.columns(2)

        with colA:
            st.write("### Source A")
            figA = plt.figure(figsize=(5, 3))
            plt.imshow(wc1, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(figA, width="stretch")
            plt.close(figA)

        with colB:
            st.write("### Source B")
            figB = plt.figure(figsize=(5, 3))
            plt.imshow(wc2, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(figB, width="stretch")
            plt.close(figB)

        st.success("✔ Comparison Completed!")
