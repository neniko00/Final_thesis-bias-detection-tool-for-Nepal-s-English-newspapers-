import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download once
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    """
    Performs cleaning, tokenization, and lemmatization
    """

    # Remove URLs and HTML
    text = re.sub(r"http\S+|www\S+|<.*?>", "", text)

    # Lowercase
    text = text.lower()

    # Remove non-alphabetic characters
    text = re.sub(r"[^a-z\s]", "", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return " ".join(lemmatized_tokens)
