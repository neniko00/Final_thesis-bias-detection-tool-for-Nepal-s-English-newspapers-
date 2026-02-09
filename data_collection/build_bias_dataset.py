import pandas as pd
import numpy as np
import re

BIAS_KEYWORDS = [
    "shocking", "outrageous", "disaster", "corrupt", "blamed", "accused",
    "scandal", "failed", "disgrace", "exposed", "manipulated", "biased",
    "alleged", "controversial", "propaganda", "misleading",
    "critics argue", "opponents claim", "unfair", "one-sided",
]

def detect_bias_score(text):
    """Returns a simple bias-score based on presence of keywords."""
    text = text.lower()
    score = sum(1 for word in BIAS_KEYWORDS if word in text)
    return score

def build_bias_dataset():
    print("Building bias dataset...")

    # Load clean REAL Nepal news
    try:
        himalayan = pd.read_csv("dataset/himalayan_times_raw.csv")
        kathmandu = pd.read_csv("dataset/kathmandu_post_raw.csv")
        southasia = pd.read_csv("dataset/southasiacheck_raw.csv")

        real_df = pd.concat([himalayan, kathmandu, southasia], ignore_index=True)
        real_df["label"] = "non-biased"
        print("Loaded REAL Nepal news.")
    except:
        print("ERROR: Could not load Nepal news datasets.")
        real_df = pd.DataFrame(columns=["title", "text", "label"])

    # Detect bias in REAL articles (many Nepal articles have mild bias)
    real_df["bias_score"] = real_df["text"].astype(str).apply(detect_bias_score)
    real_df["label"] = real_df["bias_score"].apply(lambda x: "biased" if x > 1 else "non-biased")

    # Add randomly generated biased data using keyword injection
    synthetic_bias = []
    for i in range(1500):
        base_text = (
            "The government announced a new policy today. "
            "Critics argue that the move is unfair and controversial. "
        )
        synthetic_bias.append([f"Synthetic Bias Article {i}", base_text, "biased"])

    synthetic_df = pd.DataFrame(synthetic_bias, columns=["title", "text", "label"])

    # Combine all
    combined = pd.concat([real_df, synthetic_df], ignore_index=True)

    # Shuffle
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    combined.to_csv("dataset/bias_dataset_raw.csv", index=False)
    print("Saved: dataset/bias_dataset_raw.csv")
    print("Total samples:", len(combined))

build_bias_dataset()
