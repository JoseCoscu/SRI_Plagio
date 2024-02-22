import os
from nltk.util import ngrams
import spacy
import nltk
import gensim
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

def get_ngrams(text, n):
    """
    Genera n-gramas de un texto dado.

    Parámetros:
    text (str): El texto del cual se generarán los n-gramas.
    n (int): El tamaño de los n-gramas.

    Retorna:
    list: Una lista de n-gramas.
    """
    # Divide el texto en tokens
    tokens = text.split()

    # Genera los n-gramas
    ngrams_list = list(ngrams(tokens, n))

    return [' '.join(gram) for gram in ngrams_list]


def find_similar_ngrams(document1, document2, n=10):
    """
    Encuentra n-gramas similares entre dos documentos.

    Parámetros:
    document1 (str): El primer documento.
    document2 (str): El segundo documento.
    n (int): El tamaño de los n-gramas (por defecto es 10).

    Retorna:
    set: Un conjunto de n-gramas comunes entre los dos documentos.
    """
    # Obtiene los n-gramas de ambos documentos
    ngrams_doc1 = get_ngrams(document1, n)
    ngrams_doc2 = get_ngrams(document2, n)

    set1 = set(ngrams_doc1)
    set2 = set(ngrams_doc2)

    # Encuentra los n-gramas comunes
    common_ngrams = set1 & set2

    return common_ngrams


def load_docs(lower=False):
    """
    Carga documentos de una carpeta llamada 'data' en el directorio actual.

    Parámetros:
    lower (bool): Indica si se deben convertir los documentos a minúsculas (por defecto es False).

    Retorna:
    list: Una lista de contenidos de documentos.
    """
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
    """
    Realiza tokenización utilizando Spacy para una lista de textos.

    Parámetros:
    texts (list): Una lista de textos a tokenizar.

    Retorna:
    list: Una lista de listas de tokens.
    """
    nlp = spacy.load("es_core_news_sm")
    return [[token for token in nlp(doc)] for doc in texts]


def remove_noise_spacy(tokenized_docs):
    """
    Elimina tokens no alfabéticos de textos tokenizados utilizando Spacy.

    Parámetros:
    tokenized_docs (list): Una lista de listas de tokens.

    Retorna:
    list: Una lista de listas de tokens limpios.
    """
    return [[token for token in doc if token.is_alpha] for doc in tokenized_docs]


def remove_stopwords_spacy(tokenized_docs):
    """
    Elimina stopwords de textos tokenizados utilizando Spacy.

    Parámetros:
    tokenized_docs (list): Una lista de listas de tokens.

    Retorna:
    list: Una lista de listas de tokens sin stopwords.
    """
    stopwords = spacy.lang.es.stop_words.STOP_WORDS
    return [
        [token for token in doc if token.text not in stopwords] for doc in tokenized_docs
    ]


def morphological_reduction_spacy(tokenized_docs, use_lemmatization=True):
    """
    Realiza reducción morfológica en textos tokenizados utilizando Spacy.

    Parámetros:
    tokenized_docs (list): Una lista de listas de tokens.
    use_lemmatization (bool): Indica si se debe usar lematización (por defecto es True).

    Retorna:
    list: Una lista de listas de tokens reducidos morfológicamente.
    """
    stemmer = nltk.stem.PorterStemmer()
    return [
        [token.lemma_ if use_lemmatization else stemmer.stem(token.text) for token in doc]
        for doc in tokenized_docs
    ]


def filter_tokens_by_occurrence(tokenized_docs, no_below=2, no_above=10):
    """
    Filtra tokens en base a su frecuencia de ocurrencia en una colección de documentos.

    Parámetros:
    tokenized_docs (list): Una lista de listas de tokens.
    no_below (int): Frecuencia mínima de ocurrencia para conservar un token (por defecto es 2).
    no_above (int): Frecuencia máxima de ocurrencia para conservar un token (por defecto es 10).

    Retorna:
    tuple: Una tupla que contiene una lista de listas de tokens filtrados y el diccionario resultante.
    """
    dictionary = gensim.corpora.Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below, no_above)

    filtered_words = [word for _, word in dictionary.iteritems()]
    filtered_tokens = [
        [word for word in doc if word in filtered_words]
        for doc in tokenized_docs
    ]

    return filtered_tokens, dictionary


def build_vocabulary(dictionary):
    """
    Construye un vocabulario a partir de un diccionario.

    Parámetros:
    dictionary (gensim.corpora.Dictionary): Un diccionario de términos.

    Retorna:
    list: Una lista de términos del vocabulario.
    """
    vocabulary = list(dictionary.token2id.keys())
    return vocabulary


def vector_representation(tokenized_docs, dictionary, use_bow=True):
    """
    Genera representaciones vectoriales de documentos utilizando modelos de bolsa de palabras o TF-IDF.

    Parámetros:
    tokenized_docs (list): Una lista de listas de tokens.
    dictionary (gensim.corpora.Dictionary): El diccionario utilizado para la representación.
    use_bow (bool): Indica si se debe usar el modelo de bolsa de palabras (por defecto es True).

    Retorna:
    list: Una lista de representaciones vectoriales de documentos.
    """
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    if use_bow:
        vector_repr = corpus
    else:
        tfidf = gensim.models.TfidfModel(corpus)
        vector_repr = [tfidf[doc] for doc in corpus]

    return vector_repr


def v_similarity(v1, v2):
    """
    Calcula la similitud de coseno entre dos vectores.

    Parámetros:
    v1 (list): El primer vector.
    v2 (list): El segundo vector.

    Retorna:
    float: La similitud de coseno entre los dos vectores.
    """
    len_diff = len(v2) - len(v1)
    if len_diff > 0:
        v1.extend([0] * len_diff)
    elif len_diff < 0:
        v2.extend([0] * abs(len_diff))

    # Calcula la similitud de coseno entre los dos vectores
    similarity = cosine_similarity([v1], [v2])
    return similarity[0][0]


def extracting_vectors(v_repr):
    """
    Extrae los vectores de una representación vectorial.

    Parámetros:
    v_repr (list): Una lista de representaciones vectoriales de documentos.

    Retorna:
    list: Una lista de vectores.
    """
    vectors = [[x[1] for x in docs] for docs in v_repr]
    return vectors


def get_doc_embedding(doc_tokens, model):
    """
    Obtiene el embedding de un documento a partir de sus tokens y un modelo de Word2Vec.

    Parámetros:
    doc_tokens (list): Una lista de tokens del documento.
    model (gensim.models.Word2Vec): Un modelo de Word2Vec.

    Retorna:
    list: El embedding del documento.
    """
    doc_embedding = []
    for token in doc_tokens:
        if token in model.wv:
            doc_embedding.append(model.wv[token])
    if not doc_embedding:
        # Si no hay embeddings para el documento, retorna un vector de ceros
        return [0] * model.vector_size
    # Promediar los embeddings de las palabras para obtener el embedding del documento
    return sum(doc_embedding) / len(doc_embedding)
