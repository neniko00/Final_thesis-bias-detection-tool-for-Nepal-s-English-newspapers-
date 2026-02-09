import pandas as pd

def build_fake_news_dataset():
    print("Building fake news dataset...")

    # ISOT dataset
    try:
        df_fake = pd.read_csv("https://raw.githubusercontent.com/diptamath/covid_fake_news/main/Fake.csv")
        df_fake = df_fake.rename(columns={"text": "text"})
        df_fake["label"] = "fake"
    except:
        df_fake = pd.DataFrame(columns=["text", "label"])

    # LIAR dataset (only false statements)
    try:
        liar = pd.read_csv("https://raw.githubusercontent.com/EthanBites/LIAR-Dataset/main/data/liar_test.csv")
        liar = liar[liar["label"] == "false"]
        liar["text"] = liar["statement"]
        liar = liar[["text"]]
        liar["label"] = "fake"
    except:
        liar = pd.DataFrame(columns=["text", "label"])

    combined = pd.concat([df_fake, liar], ignore_index=True)
    combined.to_csv("dataset/fake_news_raw.csv", index=False)
    print("Saved fake news raw dataset!")

build_fake_news_dataset()
