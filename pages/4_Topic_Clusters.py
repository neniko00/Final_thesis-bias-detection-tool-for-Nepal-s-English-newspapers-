import streamlit as st
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud

st.title("🧩 Topic Clustering (Unsupervised NLP)")

st.write(
    "This module groups the input text into multiple topics using "
    "**Latent Dirichlet Allocation (LDA)**."
)

# ----------------------------------
# CLEAN TEXT FUNCTION
# ----------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


text_input = st.text_area(
    "Paste your article text here:",
    height=200
)

if st.button("Generate Topic Clusters"):
    if not text_input.strip():
        st.error("Please enter article text.")
    else:
        cleaned_text = clean_text(text_input)

        # ----------------------------------
        # TF-IDF VECTORIZATION
        # ----------------------------------
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform([cleaned_text])

        # ----------------------------------
        # LDA TOPIC MODEL
        # ----------------------------------
        lda = LatentDirichletAllocation(
            n_components=3,
            random_state=42
        )
        lda.fit(X)

        st.subheader("📌 Topic Distribution")

        topic_scores = lda.transform(X)[0]
        for i, score in enumerate(topic_scores):
            st.write(f"**Topic {i + 1}:** {score:.3f}")

        # ----------------------------------
        # WORD CLOUDS
        # ----------------------------------
        st.subheader("☁ WordCloud for Detected Topics")

        words = vectorizer.get_feature_names_out()
        components = lda.components_

        for topic_id, topic_values in enumerate(components):
            st.markdown(f"### Topic {topic_id + 1}")

            topic_words = {
                words[i]: topic_values[i]
                for i in range(len(words))
            }

            wc = WordCloud(
                width=600,
                height=400,
                background_color="white"
            ).generate_from_frequencies(topic_words)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")

            # ✅ FIXED: new Streamlit API (no warning)
            st.pyplot(fig, width="stretch")
