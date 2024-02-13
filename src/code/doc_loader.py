import spacy
import nltk
import gensim
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def_path = "../../data/"


def load_docs(path):
    docs = []

    # Obtener la lista de archivos en la carpeta
    archivos = os.listdir(def_path)

    # Iterar sobre cada archivo en la carpeta
    for archivo in archivos:

        if archivo.endswith('.txt') or archivo.endswith('.pdf'):
            # Construir la ruta completa al archivo
            ruta_completa = os.path.join(def_path, archivo)

            # Realizar operaciones con el archivo, por ejemplo, cargarlo
            with open(ruta_completa, 'r') as f:
                contenido = f.read()

            # Agregar el contenido del archivo a la lista
            docs.append(contenido)

    return docs


def tokenization_spacy(texts):
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
    global dictionary
    dictionary = gensim.corpora.Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below, no_above)

    filtered_words = [word for _, word in dictionary.iteritems()]
    filtered_tokens = [
        [word for word in doc if word in filtered_words]
        for doc in tokenized_docs
    ]

    return filtered_tokens


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


nlp = spacy.load("es_core_news_sm")
docs = load_docs(def_path)
tokenized_docs = tokenization_spacy(docs)
tokenized_docs = remove_noise_spacy(tokenized_docs)
tokenized_docs = remove_stopwords_spacy(tokenized_docs)
tokenized_docs = morphological_reduction_spacy(tokenized_docs)

filtered_docs = filter_tokens_by_occurrence(tokenized_docs)

vocabulary = build_vocabulary(dictionary)

vector_repr = vector_representation(tokenized_docs, dictionary)

vs = extracting_vectors(vector_repr)


s = v_similarity(vs[2],vs[3])
print(s)
