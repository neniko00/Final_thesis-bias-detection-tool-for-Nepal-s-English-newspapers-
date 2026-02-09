from nlp.preprocessing import preprocess_text
from nlp.preprocessing_spacy import preprocess_text_spacy

def preprocess_text_factory(text: str, method: str = "nltk") -> str:
    """
    Factory function to select NLP preprocessing method
    method: 'nltk' or 'spacy'
    """
    if method == "spacy":
        return preprocess_text_spacy(text)
    return preprocess_text(text)
