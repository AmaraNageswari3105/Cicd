import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Safe initialization for NLTK datasets
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

def clean_text(text: str) -> str:
    """Basic text cleaning logic for generative models."""
    if not isinstance(text, str):
        return ""
    
    # Remove large space repetitions
    text = re.sub(r'\s+', ' ', text)
    # Allow alphanumeric characters and some punctuation marks
    text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
    
    return text.strip()

def tokenize_and_remove_stopwords(text: str) -> str:
    """Tokenize the input text and remove stopwords logic."""
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if not w.lower() in stop_words]
    return " ".join(filtered_tokens)

