# Import necessary libraries
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.utils import simple_preprocess
from multiprocessing import freeze_support
from multiprocessing import process
import gensim
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')

# Preprocess text data
def preprocess_text(text):
    spanish_stopwords = set(stopwords.words('spanish'))
    return [token for token in simple_preprocess(text) if token not in gensim.parsing.preprocessing.STOPWORDS and token not in spanish_stopwords and len(token) > 3 and 'twitter.com' not in token]

if __name__ == '__main__':
    freeze_support()

    # Load data
    data = pd.read_csv("tweets_municipalidad.csv")
    documents = data['tweet'].astype(str)

    # Preprocesamiento
    processed_documents = [' '.join(preprocess_text(text)) for text in documents]

    # Convertir documentos a formato Gensim
    corpus_gensim = [doc.split() for doc in processed_documents]
    dictionary = corpora.Dictionary(corpus_gensim)
    corpus_bow = [dictionary.doc2bow(doc) for doc in corpus_gensim]
    print(corpus_bow)

    # Build LDA model
    lda_model = LdaModel(corpus=corpus_bow,
                        id2word=dictionary,
                        num_topics=15,
                        random_state=42,
                        passes=30,
                        alpha='asymmetric',
                        eta=0.9)

    # Print top 10 words for each topic
    for topic in lda_model.print_topics(num_topics=15, num_words=10):
        print(topic)

    # Calcular coherencia
    COHERENCE_MEASURES = ['u_mass', 'c_v', 'c_w2v', 'c_uci', 'c_npmi']

    for coherence_measure in COHERENCE_MEASURES:
        coherences_results = CoherenceModel(model=lda_model, texts=corpus_gensim, dictionary=dictionary, coherence=coherence_measure)
        coherences_results = coherences_results.get_coherence()
        print(f'\nCoherencia {coherence_measure}: {coherences_results}')

    # Calcular la perplexity
    print(np.exp2(-lda_model.log_perplexity(corpus_bow))) #https://stackoverflow.com/questions/55278701/gensim-topic-modeling-with-mallet-perplexity
    