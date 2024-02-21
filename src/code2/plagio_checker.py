from doc_loader import *


def process_docs():
    docs = load_docs(True)

    # print("N-gramas comunes:")
    # for ngram in common_ngrams:
    #     print(ngram)

    tokenized_docs = tokenization_spacy(docs)
    tokenized_docs = remove_noise_spacy(tokenized_docs)
    tokenized_docs = remove_stopwords_spacy(tokenized_docs)
    tokenized_docs = [[token.text for token in t_docs] for t_docs in tokenized_docs]
    model = Word2Vec(tokenized_docs, vector_size=100, window=5, min_count=1, sg=0)
    embeddings = [get_doc_embedding(doc, model) for doc in tokenized_docs]
    return embeddings


def compare_docs(v_doc1, v_doc2):
    s = v_similarity(v_doc1, v_doc2)
    return s


def find_n_grams(doc1, doc2, n):
    common_ngrams = find_similar_ngrams(doc1, doc2, n)
    return common_ngrams
