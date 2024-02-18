from doc_loader import *


def process_docs():

    docs = load_docs(True)

    # print("N-gramas comunes:")
    # for ngram in common_ngrams:
    #     print(ngram)

    tokenized_docs = tokenization_spacy(docs)
    tokenized_docs = remove_noise_spacy(tokenized_docs)
    tokenized_docs = remove_stopwords_spacy(tokenized_docs)
    tokenized_docs = morphological_reduction_spacy(tokenized_docs)

    filtered_docs, dictionary = filter_tokens_by_occurrence(tokenized_docs)

    vector_repr = vector_representation(tokenized_docs, dictionary)

    vs = extracting_vectors(vector_repr)

    return vs


def compare_docs(v_doc1, v_doc2):
    s = v_similarity(v_doc1, v_doc2)
    return s


def find_n_grams(doc1, doc2, n):
    common_ngrams = find_similar_ngrams(doc1, doc2, n)
    return common_ngrams


