"""
Global configuration settings for BiasDetectionTool
"""

# -------------------------------------------------
# NLP preprocessing engine
# -------------------------------------------------
# Options:
#   "nltk"  → NLTK-based preprocessing
#   "spacy" → spaCy-based preprocessing
NLP_ENGINE = "spacy"

# -------------------------------------------------
# Explainability engine
# -------------------------------------------------
# Options:
#   "tfidf" → TF-IDF feature importance
#   "shap"  → SHAP explainable AI
EXPLAINABILITY_ENGINE = "shap"

# -------------------------------------------------
# Application-level settings (future safe)
# -------------------------------------------------
APP_NAME = "Bias Detection Tool"
APP_VERSION = "1.0"
DEBUG_MODE = False
