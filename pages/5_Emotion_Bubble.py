import streamlit as st
from nrclex import NRCLex
import pandas as pd
import matplotlib.pyplot as plt

st.title("🎭 Emotion Bubble Chart")
st.write("This module visualizes emotional intensities present in the text using a bubble chart.")

# ------------------------------
# Input Text
# ------------------------------
text_input = st.text_area("Paste your article text here:")

# ------------------------------
# Process Button
# ------------------------------
if st.button("Generate Emotion Chart"):

    if not text_input.strip():
        st.error("Please enter some text first.")
    else:
        # ------------------------------
        # Extract emotion scores
        # ------------------------------
        emotion = NRCLex(text_input)
        emotion_scores = emotion.raw_emotion_scores

        if not emotion_scores:
            st.warning("No emotions detected in the text.")
        else:
            st.subheader("🧠 Emotion Scores Detected")
            st.json(emotion_scores)

            # Convert to DataFrame
            df = pd.DataFrame(
                emotion_scores.items(),
                columns=["Emotion", "Score"]
            )

            # ------------------------------
            # Bubble Chart
            # ------------------------------
            st.subheader("🎈 Emotion Bubble Chart")

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.scatter(
                df["Emotion"],
                df["Score"],
                s=df["Score"] * 800,   # bubble size
                alpha=0.6,
                color="skyblue",
                edgecolors="black"
            )

            for _, row in df.iterrows():
                ax.text(
                    row["Emotion"],
                    row["Score"],
                    row["Emotion"],
                    fontsize=10,
                    ha="center"
                )

            ax.set_xlabel("Emotion Type")
            ax.set_ylabel("Intensity")
            ax.set_title("Emotion Bubble Visualization")
            plt.xticks(rotation=45)
            plt.tight_layout()

            # ✅ Correct Streamlit call (no warning)
            st.pyplot(fig, width="stretch")

            # ------------------------------
            # Summary
            # ------------------------------
            st.subheader("📌 Summary of Emotional Content")
            st.write("""
            - Larger bubbles indicate stronger emotional presence.  
            - Negative emotions (anger, fear, disgust) often correlate with bias.  
            - Positive emotions (joy, trust, anticipation) often reduce perceived bias.  
            """)
