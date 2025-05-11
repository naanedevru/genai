import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

model = api.load("word2vec-google-news-300")

def get_similar_words(word, topn=5):
    return [w[0] for w in model.similar_by_word(word, topn=topn)] if word in model.key_to_index else f"Word '{word}' not found."

print(f"Top 5 similar words to 'computer': {get_similar_words('computer')}")

tech_words = ["computer", "algorithm", "software", "hardware", "data", "network", "AI", "robotics", "internet", "cybersecurity"]
word_vectors = np.array([model[word] for word in tech_words])

pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(word_vectors)

tsne = TSNE(n_components=3, perplexity=5, random_state=42)
tsne_vectors = tsne.fit_transform(word_vectors)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
for i, word in enumerate(tech_words):
    ax.scatter(tsne_vectors[i, 0], tsne_vectors[i, 1], tsne_vectors[i, 2])
    ax.text(tsne_vectors[i, 0], tsne_vectors[i, 1], tsne_vectors[i, 2], word, fontsize=12)
ax.set_title("3D t-SNE Projection of Word Embeddings")
plt.show()
