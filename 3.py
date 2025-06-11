PG 3
!pip install gensim matplotlib
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import STOPWORDS
legal_corpus = [ 'Contracts are binding agreements', 'The plaintiff is alleging fraud', 'The defendant was found guilty', 'Tort law covers negligence', 'Litigation is still ongoing', 'Jurisdiction is exclusive in some cases', 'Arbitration is legally binding', 'A subpoena is mandatory to appear in court', 'An affidavit is a sworn statement',
    'Strict liability applies in certain cases'  ]
legal_docs = [[word for word in sentence.lower().split() if word not in STOPWORDS] for sentence in legal_corpus]
model = Word2Vec(sentences=legal_docs, vector_size=3, window=5, min_count=1)
print("Vocabulary:", model.wv.index_to_key)
word = 'contracts'
if word in model.wv:
    print(f"Vector for '{word}':", model.wv[word])
    print(f"Top 5 similar words to '{word}':", model.wv.most_similar(word, topn=5))
else:
    print(f"'{word}' not found in vocabulary.")
