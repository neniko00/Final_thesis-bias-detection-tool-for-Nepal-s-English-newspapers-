import pandas as pd
import re

def clean(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s.,']", " ", text)
    text = text.lower()
    return text

def clean_all():
    files = [
        "himalayan_times_raw.csv",
        "kathmandu_post_raw.csv",
        "southasiacheck_raw.csv",
        "fake_news_raw.csv",
    ]

    final = []

    for f in files:
        try:
            df = pd.read_csv("dataset/" + f)
            df["clean_text"] = df["text"].astype(str).apply(clean)
            final.append(df)
            print(f"Cleaned: {f}")
        except:
            print(f"Failed to clean: {f}")
            continue

    final_df = pd.concat(final, ignore_index=True)
    final_df.to_csv("dataset/final_dataset.csv", index=False)
    print("Saved final_dataset.csv!")

clean_all()
