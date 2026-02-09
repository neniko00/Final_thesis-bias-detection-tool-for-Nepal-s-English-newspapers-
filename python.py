import nltk
import pandas as pd
df = pd.read_csv("dataset/bias_dataset.csv")
print(df.columns)
nltk.download('punkt')
nltk.download('wordnet')
exit()
