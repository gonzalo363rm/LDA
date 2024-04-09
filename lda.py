import pandas as pd
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.coherencemodel import CoherenceModel
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from unidecode import unidecode
import spacy
from spacy.lang.es import Spanish
import nltk
from multiprocessing import freeze_support  # Agregado

# Agregar esta línea para evitar el error
if __name__ == '__main__':
    freeze_support()

    # Resto del código aquí
    nlp = spacy.load('es_core_news_sm')

    # Otro código aquí

    spanish_stopwords = set(stopwords.words('spanish'))

    def preprocess(text):
        result = []
        doc = nlp(text)
        for token in doc:
            if token.is_alpha and token.text.lower() not in spanish_stopwords and len(token.text) > 3:
                token_text = unidecode(token.text.lower())
                result.append(token_text)
        return result

    docs = pd.read_csv('C:/Users/gonza/Desktop/J/wFacu/Tesina/LDA/tweets_municipalidad.csv')
    docs['tweet'] = docs['tweet'].astype(str)
    processed_data = docs['tweet'].map(preprocess)

    dictionary = corpora.Dictionary(processed_data)
    corpus = [dictionary.doc2bow(text) for text in processed_data]

    num_topics = 8
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, alpha=(50/num_topics), eta=0.5)

    # Calcular coherencia
    coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_data, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()

    print(f'\nCoherencia del modelo LDA: {coherence_lda}')

    topics = lda_model.print_topics(num_topics=num_topics)
    for topic in topics:
        print(topic)