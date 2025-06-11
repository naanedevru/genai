PG 2
import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
model = api.load("word2vec-google-news-300")
def get_similar_words(word, topn=5):
    if word in model.key_to_index:
        return [w for w, _ in model.most_similar(word, topn=topn)]
    return f"Word '{word}' not found."
print("Top 5 similar words to 'computer':", get_similar_words('computer'))
tech_words = ["computer", "algorithm", "software", "hardware", "data", "network", "AI", "robotics", "internet", "cybersecurity"]
word_vectors = np.array([model[word] for word in tech_words])
tsne_vectors = TSNE(n_components=3, perplexity=5, random_state=42).fit_transform(word_vectors)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D t-SNE Projection of Tech Word Embeddings")
for i, word in enumerate(tech_words):
    x, y, z = tsne_vectors[i]
    ax.scatter(x, y, z)
    ax.text(x, y, z, word, fontsize=10)
plt.show()
