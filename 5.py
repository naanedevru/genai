import nltk
import gensim.downloader as api
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download(['punkt', 'stopwords'])

print("Loading pre-trained word vectors...")
word_vectors = api.load("glove-wiki-gigaword-100")
print("Word vectors loaded successfully!")

def get_similar_words(seed_word, top_n=5):
    try:
        return [word[0] for word in word_vectors.most_similar(seed_word, topn=top_n)]
    except KeyError:
        print(f"'{seed_word}' not found in vocabulary.")
        return []

def generate_sentence(seed_word, similar_words):
    if len(similar_words) < 5:
        return f"The word '{seed_word}' is unique, making it difficult to find similar words."

    sentence_templates = [
        f"The {seed_word} was surrounded by {similar_words[0]} and {similar_words[1]}.",
        f"People often associate {seed_word} with {similar_words[2]} and {similar_words[3]}.",
        f"In the land of {seed_word}, {similar_words[4]} was a common sight.",
        f"A story about {seed_word} would be incomplete without {similar_words[1]} and {similar_words[3]}.",
    ]
    return random.choice(sentence_templates)

def generate_paragraph(seed_word):
    similar_words = get_similar_words(seed_word)
    if not similar_words:
        return None
    return " ".join([generate_sentence(seed_word, similar_words) for _ in range(4)])

while True:
    seed_word = input("Enter a seed word: ").strip().lower()
    paragraph = generate_paragraph(seed_word)
    if paragraph:
        print("\nGenerated Paragraph:\n", paragraph)
        break
    else:
        print("\nPlease try again with another word.\n")
