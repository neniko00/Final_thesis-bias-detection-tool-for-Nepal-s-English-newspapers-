import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

def preprocess_text_spacy(text: str) -> str:
    """
    Preprocess text using spaCy:
    - Lowercasing
    - Tokenization
    - Lemmatization
    - Stopword and punctuation removal
    """
    if not text:
        return ""

    doc = nlp(text.lower())

    tokens = [
        token.lemma_
        for token in doc
        if token.is_alpha and not token.is_stop
    ]

    return " ".join(tokens)
