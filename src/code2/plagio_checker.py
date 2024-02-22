from doc_loader import *

def process_docs():
    """
    Procesa documentos cargados utilizando diversas técnicas de procesamiento de lenguaje natural.

    Retorna:
    list: Una lista de embeddings de documentos.
    """
    docs = load_docs(True)

    tokenized_docs = tokenization_spacy(docs)
    tokenized_docs = remove_noise_spacy(tokenized_docs)
    tokenized_docs = remove_stopwords_spacy(tokenized_docs)
    tokenized_docs = [[token.text for token in t_docs] for t_docs in tokenized_docs]
    model = Word2Vec(tokenized_docs, vector_size=100, window=5, min_count=1, sg=0)
    embeddings = [get_doc_embedding(doc, model) for doc in tokenized_docs]
    return embeddings


def compare_docs(v_doc1, v_doc2):
    """
    Compara la similitud entre dos documentos basados en sus vectores de embedding.

    Parámetros:
    v_doc1 (list): El vector de embedding del primer documento.
    v_doc2 (list): El vector de embedding del segundo documento.

    Retorna:
    float: La similitud entre los dos documentos.
    """
    s = v_similarity(v_doc1, v_doc2)
    return s


def find_n_grams(doc1, doc2, n):
    """
    Encuentra n-gramas comunes entre dos documentos.

    Parámetros:
    doc1 (str): El primer documento.
    doc2 (str): El segundo documento.
    n (int): El tamaño de los n-gramas.

    Retorna:
    set: Un conjunto de n-gramas comunes entre los dos documentos.
    """
    common_ngrams = find_similar_ngrams(doc1, doc2, n)
    return common_ngrams
