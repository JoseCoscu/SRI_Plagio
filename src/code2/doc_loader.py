import spacy
import nltk
import gensim
import os
from sklearn.metrics.pairwise import cosine_similarity
from nltk.util import ngrams
import numpy as np

def_path = os.path


def get_ngrams(text, n):
    # Divide el texto en tokens
    tokens = text.split()

    # Genera los n-gramas
    ngrams_list = list(ngrams(tokens, n))

    return [' '.join(gram) for gram in ngrams_list]


def find_similar_ngrams(document1, document2, n=10):
    # Obtiene los n-gramas de ambos documentos
    ngrams_doc1 = get_ngrams(document1, n)
    ngrams_doc2 = get_ngrams(document2, n)

    set1 = set(ngrams_doc1)
    set2 = set(ngrams_doc2)

    # Encuentra los n-gramas comunes
    common_ngrams = set1 & set2

    return common_ngrams


def load_docs(lower):
    docs = []
    path = os.getcwd()
    path = os.path.join(path, 'data')
    # Obtener la lista de archivos en la carpeta
    archivos = os.listdir(path)

    # Iterar sobre cada archivo en la carpeta
    for archivo in archivos:

        if archivo.endswith('.txt') or archivo.endswith('.pdf'):
            # Construir la ruta completa al archivo
            ruta_completa = os.path.join(path, archivo)

            # Realizar operaciones con el archivo, por ejemplo, cargarlo
            with open(ruta_completa, 'r', encoding='utf-8') as f:
                contenido = f.read()

            # Agregar el contenido del archivo a la lista
            docs.append(contenido.lower() if lower else contenido)

    return docs


def tokenization_spacy(texts):
    nlp = spacy.load("es_core_news_sm")
    return [[token for token in nlp(doc)] for doc in texts]


def remove_noise_spacy(tokenized_docs):
    return [[token for token in doc if token.is_alpha] for doc in tokenized_docs]


def remove_stopwords_spacy(tokenized_docs):
    stopwords = spacy.lang.es.stop_words.STOP_WORDS
    return [
        [token for token in doc if token.text not in stopwords] for doc in tokenized_docs
    ]


def morphological_reduction_spacy(tokenized_docs, use_lemmatization=True):
    stemmer = nltk.stem.PorterStemmer()
    return [
        [token.lemma_ if use_lemmatization else stemmer.stem(token.text) for token in doc]
        for doc in tokenized_docs
    ]


def filter_tokens_by_occurrence(tokenized_docs, no_below=2, no_above=10):
    dictionary = gensim.corpora.Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below, no_above)

    filtered_words = [word for _, word in dictionary.iteritems()]
    filtered_tokens = [
        [word for word in doc if word in filtered_words]
        for doc in tokenized_docs
    ]

    return filtered_tokens, dictionary


def build_vocabulary(dictionary):
    vocabulary = list(dictionary.token2id.keys())
    return vocabulary


def vector_representation(tokenized_docs, dictionary, use_bow=False):
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    if use_bow:
        vector_repr = corpus
    else:
        tfidf = gensim.models.TfidfModel(corpus)
        vector_repr = [tfidf[doc] for doc in corpus]

    return vector_repr


def v_similarity(v1, v2):
    len_diff = len(v2) - len(v1)
    if len_diff > 0:
        v1.extend([0] * len_diff)
    elif len_diff < 0:
        v2.extend([0] * abs(len_diff))

    # Calcula la similitud de coseno entre los dos vectores
    similarity = cosine_similarity([v1], [v2])
    return similarity[0][0]


def extracting_vectors(v_repr):
    vectors = [[x[1] for x in docs] for docs in v_repr]
    return vectors


def find_differing_indices(vector1, vector2, alpha):
    differing_indices = []

    # Iterar sobre las componentes de los vectores y comparar sus valores
    for i, (value1, value2) in enumerate(zip(vector1, vector2)):
        dif = abs(value1 - value2)
        if dif <= alpha:
            differing_indices.append(i)

    return differing_indices


