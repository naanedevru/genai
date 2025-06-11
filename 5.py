PG 5
import nltk
import gensim.downloader as api
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download(['punkt', 'stopwords'])
print("Loading pre-trained word vectors...")
word_vectors = api.load("glove-wiki-gigaword-100")
print("Word vectors loaded successfully!")
def get_similar_words(seed, top_n=5):
    try:
        return [word[0] for word in word_vectors.most_similar(seed, topn=top_n)]
    except KeyError:
        print(f"'{seed}' not found in vocabulary.")
        return []
def generate_sentence(seed, similar):
    if len(similar) < 5:
        return f"The word '{seed}' is unique, making it difficult to find similar words."

    sentence_templates = [
        f"The {seed} was surrounded by {similar[0]} and {similar[1]}.",
        f"People often associate {seed} with {similar[2]} and {similar[3]}.",
        f"In the land of {seed}, {similar[4]} was a common sight.",
        f"A story about {seed} would be incomplete without {similar[1]} and {similar[3]}.",    ]
    return random.choice(sentence_templates)
def generate_paragraph(seed):
    similar = get_similar_words(seed)
    if not similar:
        return None
    return " ".join([generate_sentence(seed, similar) for _ in range(4)])
while True:
    seed = input("Enter a seed word: ").strip().lower()
    paragraph = generate_paragraph(seed)
    if paragraph:
        print("\nGenerated Paragraph:\n", paragraph)
        break
    else:
        print("\nPlease try again with another word.\n")
