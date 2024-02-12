import spacy
import nltk
import gensim

def_path = "../../data/"


def fix_doc(doc):
    f_doc = []
    for i in doc:
        f_doc.append(i[0])
    return f_doc


def load_doc(name):
    # Abre el archivo en modo lectura ('r')
    with open(def_path + name, 'r') as archivo:
        # Lee el contenido del archivo
        doc = archivo.read()

    # Imprime el contenido del archivo
    return doc


def tokenization_spacy(texts):
    return [[token for token in nlp(doc)] for doc in texts]


def remove_empty(words):
    while True:
        try:
            # print('re')
            words.remove([])
        except:
            break

    return words


def remove_noise_spacy(tokenized_docs):
    return [[token for token in doc if token.is_alpha] for doc in tokenized_docs]


def remove_stopwords_spacy(tokenized_docs):
    stopwords = spacy.lang.es.stop_words.STOP_WORDS
    return [
        [token for token in doc if token.text not in stopwords] for doc in tokenized_docs
    ]


def morphological_reduction_spacy(doc, use_lemmatization=True):
    stemmer = nltk.stem.PorterStemmer()
    return [token.lemma_ if use_lemmatization else stemmer.stem(token.text) for token in doc]


def build_vocabulary(dictionary):
    vocabulary = list(dictionary.token2id.keys())
    return vocabulary


doc1 = load_doc("naturaleza.txt").split()
doc2 = load_doc("naturaleza_plagio.txt").split()

nlp = spacy.load("es_core_news_sm")

t_doc1 = tokenization_spacy(doc1)
t_doc2 = tokenization_spacy(doc2)

t_doc1 = remove_noise_spacy(t_doc1)
t_doc2 = remove_noise_spacy(t_doc2)

t_doc1 = remove_stopwords_spacy(t_doc1)
t_doc2 = remove_stopwords_spacy(t_doc2)

remove_empty(t_doc1)
remove_empty(t_doc2)

t_doc1 = fix_doc(t_doc1)
t_doc2 = fix_doc(t_doc2)

t_doc1 = morphological_reduction_spacy(t_doc1, True)
t_doc1 = tokenization_spacy(t_doc1)
t_doc1 = remove_stopwords_spacy(t_doc1)
remove_empty(t_doc1)
t_doc1 = fix_doc(t_doc1)

v = build_vocabulary(t_doc1)


print(v)
