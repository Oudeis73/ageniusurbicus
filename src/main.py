import matplotlib.pyplot as plt
from nlp.nlp_utils import preprocess_text, get_word_freq

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

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

def main():
    file_path = "/Users/nicolalettieri/Documents/GitHub/ageniusurbicus/data/Sample_text.txt"
    raw_text = load_text(file_path)
    processed_text = preprocess_text(raw_text)
    print(f"Numero di token dopo il preprocessamento: {len(processed_text)}")
    print("Prime 10 parole preprocessate:", processed_text[:10])
    
    word_freq = get_word_freq(processed_text)
    print("\nParole più frequenti:")
    for word, count in word_freq:
        print(f"{word}: {count}")
    
    plot_word_freq(word_freq)

if __name__ == "__main__":
    main()