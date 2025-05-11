!pip install gensim matplotlib

from gensim.models import Word2Vec
from gensim.parsing.preprocessing import STOPWORDS

legal_corpus = [
    'Contracts are binding agreements', 'The plaintiff is alleging fraud', 'The defendant was found guilty',
    'Tort law covers negligence', 'Litigation is still ongoing', 'Jurisdiction is exclusive in some cases',
    'Arbitration is legally binding', 'A subpoena is mandatory to appear in court', 'An affidavit is a sworn statement',
    'Strict liability applies in certain cases'
]

legal_docs = [[word for word in sentence.lower().split() if word not in STOPWORDS] for sentence in legal_corpus]

legal_model = Word2Vec(sentences=legal_docs, vector_size=3, window=5, min_count=1)

print("Vocabulary:", legal_model.wv.index_to_key)

if 'contracts' in legal_model.wv:
    print("Vector for 'contracts':", legal_model.wv['contracts'])
    print("Similar words to 'contracts':", legal_model.wv.most_similar('contracts', topn=5))
else:
    print("'contracts' not found in vocabulary")
