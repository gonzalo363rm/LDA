# Import necessary libraries
import pandas as pd
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from gensim.models.coherencemodel import CoherenceModel

from multiprocessing import freeze_support

from nltk.corpus import stopwords

if __name__ == '__main__':
    freeze_support()

    # Preprocess text data
    def preprocess_text(text):
        spanish_stopwords = set(stopwords.words('spanish'))
        return [token for token in simple_preprocess(text) if token not in gensim.parsing.preprocessing.STOPWORDS and token not in spanish_stopwords and len(token) > 3]

    from multiprocessing import process
    # Load data
    data = pd.read_csv("tweets_municipalidad.csv")
    data = data['tweet'].astype(str)
    processed_data= [preprocess_text(text) for text in data]

    # Create dictionary and corpus
    dictionary = corpora.Dictionary(processed_data)
    corpus = [dictionary.doc2bow(text) for text in processed_data]

    # TODO: Agregar en el preprocesamiento eliminador de enlace (twitter)

    # Build LDA model
    lda_model = LdaModel(corpus=corpus,
                        id2word=dictionary,
                        num_topics=8,
                        random_state=42,
                        update_every=1,
                        chunksize=25000,
                        passes=100,
                        alpha=0.05,
                        eta=0.05,
                        per_word_topics=False)

    # Print top 10 words for each topic
    for topic in lda_model.print_topics(num_topics=8, num_words=10):
        print(topic)

    # Calcular coherencia
    coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_data, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()

    print(f'\nCoherencia del modelo LDA: {coherence_lda}')