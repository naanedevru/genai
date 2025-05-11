!pip install openai langchain langchain_community cohere nltk gensim

import openai, cohere, gensim.downloader as api, nltk, string, warnings
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

warnings.filterwarnings('ignore')

nltk.download(["punkt", "stopwords", 'punkt_tab'])

COHERE_API_KEY = 'EeKZEJ0dP0Ot2WPkcNSk1ax9p0GTrUOsfBSevUBX'  # Replace securely
co = cohere.Client(COHERE_API_KEY)
word_vectors = api.load("glove-wiki-gigaword-50")

def get_similar_words(prompt, top_n=3):
    words = [word for word in word_tokenize(prompt.lower()) if word not in stopwords.words("english") and word not in string.punctuation]
    enriched_words = [tup[0] for word in words if word in word_vectors.key_to_index for tup in word_vectors.most_similar(word, topn=top_n)]
    return " ".join(set(enriched_words))

def generate_cohere_response(prompt):
    try:
        response = co.generate(model="command", prompt=prompt, max_tokens=200, temperature=0.7)
        return response.generations[0].text.strip()
    except Exception as e:
        print(f"Error: {e}")
        return None

original_prompt = "Describe the future of artificial intelligence in education."
enriched_prompt = original_prompt + " " + get_similar_words(original_prompt)

original_response = generate_cohere_response(original_prompt)
enriched_response = generate_cohere_response(enriched_prompt)

print("\nOriginal Prompt:", original_prompt)
print("\nGenerated Response (Original Prompt):\n", original_response)
print("\nEnriched Prompt:", enriched_prompt)
print("\nGenerated Response (Enriched Prompt):\n", enriched_response)
