import streamlit as st
import re
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ---------------------------------------
# Load biased keywords & phrases
# ---------------------------------------
def load_list(file):
    try:
        with open(file, "r", encoding="utf-8") as f:
            return [w.strip().lower() for w in f.readlines()]
    except Exception:
        return []

keywords = load_list("keywords.txt")
phrases = load_list("phrases.txt")

# ---------------------------------------
# Word Bias Scoring
# ---------------------------------------
def score_word(word):
    word = word.lower()
    if word in keywords:
        return 1.0            # high bias
    elif any(p in word for p in phrases):
        return 1.0
    elif len(word) > 6:
        return 0.2            # slightly descriptive
    else:
        return 0.05           # neutral

# ---------------------------------------
# Heatmap Visualization
# ---------------------------------------
def create_heatmap(words, scores):
    # Use constrained_layout instead of tight_layout
    fig, ax = plt.subplots(
        figsize=(max(12, len(words) * 0.35), 2),
        constrained_layout=True
    )

    cmap = LinearSegmentedColormap.from_list(
        "bias_colors",
        ["#ffffff", "#ffe6e6", "#ff9999", "#ff4d4d", "#cc0000"],
        N=256
    )

    heatmap = np.array([scores])
    ax.imshow(heatmap, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=90, fontsize=8)

    ax.set_yticks([])
    ax.set_title("Bias Heatmap (Red = Strong Bias)", fontsize=14, pad=20)

    return fig

# ---------------------------------------
# Streamlit UI
# ---------------------------------------
st.title("🔥 Word-Level Bias Heatmap")
st.write("This visualization highlights each word based on its bias intensity.")

text_input = st.text_area(
    "Paste your article text here:",
    height=200
)

if st.button("Generate Heatmap"):
    if not text_input.strip():
        st.error("Please enter text.")
        st.stop()

    # Tokenize text
    words = re.findall(r"\b\w+\b", text_input)
    scores = [score_word(w) for w in words]

    # Display token → score table
    st.subheader("Token Scores")
    for w, s in zip(words, scores):
        st.write(f"**{w}** → {s}")

    # Generate heatmap
    st.subheader("🔴 Heatmap Visualization")
    fig = create_heatmap(words, scores)
    st.pyplot(fig, width="stretch")   # Streamlit future-safe

    st.success("Heatmap created successfully!")
