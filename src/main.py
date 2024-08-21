import spacy
from collections import Counter
import matplotlib.pyplot as plt

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def preprocess_text(text):
    nlp = spacy.load("la_core_web_sm")
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]

def get_word_freq(words, n=10):
    return Counter(words).most_common(n)

def plot_word_freq(word_freq):
    words, counts = zip(*word_freq)
    plt.figure(figsize=(12, 6))
    plt.bar(words, counts)
    plt.title("Parole più frequenti")
    plt.xlabel("Parole")
    plt.ylabel("Frequenza")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def get_bigrams(words):
    return list(zip(words, words[1:]))

def main():
    file_path = "data/sample_text.txt"
    raw_text = load_text(file_path)
    processed_text = preprocess_text(raw_text)
    
    print(f"Numero di token dopo il preprocessamento: {len(processed_text)}")
    print("Prime 10 parole preprocessate:", processed_text[:10])
    
    # Analisi delle frequenze
    word_freq = get_word_freq(processed_text)
    print("\nParole più frequenti:")
    for word, count in word_freq:
        print(f"{word}: {count}")
    
    # Visualizzazione
    plot_word_freq(word_freq)
    
    # Analisi dei bigrammi
    bigrams = get_bigrams(processed_text)
    bigram_freq = get_word_freq(bigrams, 5)
    print("\nBigrammi più frequenti:")
    for bigram, count in bigram_freq:
        print(f"{bigram}: {count}")

if __name__ == "__main__":
    main()