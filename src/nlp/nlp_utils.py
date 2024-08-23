import spacy
from collections import Counter

def preprocess_text(text):
    nlp = spacy.load("it_core_news_sm")
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]

def get_word_freq(words, n=10):
    return Counter(words).most_common(n)

# Funzioni aggiuntive per l'analisi NLP possono essere aggiunte qui