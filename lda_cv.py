import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.utils import simple_preprocess
from sklearn.model_selection import GridSearchCV
import numpy as np
import multiprocessing
from nltk.corpus import stopwords

# Preprocess text data
def preprocess_text(text):
    spanish_stopwords = set(stopwords.words('spanish'))
    return [token for token in simple_preprocess(text) if token not in gensim.parsing.preprocessing.STOPWORDS and token not in spanish_stopwords and len(token) > 3 and 'twitter.com' not in token]

# Esta línea es necesaria para evitar errores al usar multiprocessing en Windows
if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Cargar datos
    data = pd.read_csv("tweets_municipalidad.csv")
    documents = data['tweet'].astype(str)

    # Preprocesamiento
    processed_documents = [' '.join(preprocess_text(text)) for text in documents]

    # Crear vectorizador de palabras
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(processed_documents)
    corpus = [vectorizer.get_feature_names_out()[i] for i in range(X.shape[1])]

    # Convertir documentos a formato Gensim
    corpus_gensim = [doc.split() for doc in processed_documents]
    dictionary = corpora.Dictionary(corpus_gensim)
    corpus_bow = [dictionary.doc2bow(doc) for doc in corpus_gensim]

    # Definir parámetros para la búsqueda de hiperparámetros
    parameters = {'num_topics': [15], 'alpha': ['asymmetric'], 'eta': [0.85, 0.9, 0.95], 'passes':[15, 20, 25, 30, 35, 40, 45, 50]}

    # Realizar la búsqueda de hiperparámetros con validación cruzada
    def compute_coherence_values(corpus, dictionary, texts, num_topics, alpha, eta, passes):
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, alpha=alpha, eta=eta, passes=passes)
        coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        return coherence_model_lda.get_coherence()

    grid_search_results = []
    for num_topics in parameters['num_topics']:
        for alpha in parameters['alpha']:
            for eta in parameters['eta']:
                for passes in parameters['passes']:
                    coherence = compute_coherence_values(corpus_bow, dictionary, corpus_gensim, num_topics, alpha, eta, passes)
                    grid_search_results.append({'num_topics': num_topics, 'alpha': alpha, 'eta': eta, 'passes': passes, 'coherence': coherence})

    # Encontrar los mejores hiperparámetros
    best_params = max(grid_search_results, key=lambda x: x['coherence'])
    print("Mejores parámetros encontrados:")
    print(best_params)

    # Entrenar el modelo LDA con los mejores hiperparámetros
    best_lda_model = LdaModel(corpus=corpus_bow, id2word=dictionary, num_topics=best_params['num_topics'], alpha=best_params['alpha'], eta=best_params['eta'], passes=best_params['passes'])

    # Mostrar los tópicos aprendidos
    print("Tópicos aprendidos:")
    for idx, topic in best_lda_model.print_topics(-1):
        print(f"Tópico {idx}: {topic}")