PG 1
import gensim.downloader as api
model = api.load("word2vec-google-news-300")
queen_vector = model.get_vector("king") - model.get_vector("man") + model.get_vector("woman")
print(model.similar_by_vector(queen_vector, topn=1))
if "actor" in model.key_to_index:
    actor_vector = model.get_vector("actor") - model.get_vector("man") + model.get_vector("woman")
    print(model.similar_by_vector(actor_vector, topn=5))
else:
    print("Word 'actor' not found.")
