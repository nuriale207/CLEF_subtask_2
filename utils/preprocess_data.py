import unicodedata
#import unicode
import nltk
from numpy.compat import unicode

nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer, SnowballStemmer

nltk.download('stopwords')
#encoding: utf-8
from nltk.corpus import stopwords

nltk.download('wordnet')

import re

def stopWords(token):
    """ Aplica el stopWords de gensim a una palabra. Si la palabra es una stopWord devuelve un string vacío
        En caso contrario devuelve la palabra
                       Parámetros:
                           token -- String al que aplicar el stop words
    """

    stopwordsList=set(stopwords.words('english'))

    if token not in stopwordsList and len(token) > 3:
        return token
    else:
        return ""

def stemmer(token):
    """ Aplica el snowball stemmer de nltk a una palabra. Devuelve el String obtenido tras aplicar el stemmer
                    Parámetros:
                        token -- String al que aplicar el stemmer
    """

    stemmer = SnowballStemmer("english")


    return stemmer.stem(token)


def lemmatize(token):
    """ Aplica el lemmatizer de nltk a una palabra. Devuelve el String obtenido tras aplicar el lemmatizer
                Parámetros:
                    token -- String al que aplicar el lemmatizer
    """

    lemmatizer=WordNetLemmatizer()
    return lemmatizer.lemmatize(token)


def clean(doc):
    #return ''.join([c for c in doc if c.isalnum() or c.isspace()])
    cleaned= ''.join([c if c.isalnum() or c.isspace() else ' ' for c in doc])
    merged=' '.join([word for word in cleaned.split() if word])
    return merged


def preprocesado(docs, pStemmer, pLemmatize, pStopwords):
    """ Realiza el preprocesado del String doc aplicandole el Stemmer, lemmatizador y el stopwords

                 Parámetros:
                        docs -- String que contiene separados por espacios las palabras de los documentos sobre los que realizar el
                                preproceso.
                        pStemmer -- True si se quiere aplicar el stemmer, False en otro caso
                        pLemmatize -- True si se quiere aplicar el lemmatizer, False en otro caso
                        pStopwords -- True si se quiere aplicar el stopwords, False en otro caso
    """
    i = 0
    preprocessed_docs = []
    for text in docs:
        text = clean(text)
        filterText = []
        tokens = text.split(' ')
        for token in tokens:
            if (pStopwords == True):
                token = stopWords(token)

            if (pLemmatize == True and token != ""):
                token = lemmatize(token)

            if (pStemmer == True and token != ""):
                token = stemmer(token)
            if (token.isdigit()):
                token=""
            if (token != ""):
                # token = unidecode.unidecode(token)
                try:
                    token = unicode(token, 'utf-8')
                except (TypeError, NameError):  # unicode is a default on python 3
                    pass
                token = unicodedata.normalize('NFD', token)
                token = token.encode('ascii', 'ignore')
                token = token.decode("utf-8")
                if not token.isdigit():
                    filterText.append(token)

        i = i + 1
        if i%100==0:
            print("Procesado documento: " + str(i))

        preprocessed_docs.append(filterText)

    return preprocessed_docs


# def stopWordsDocument(doc):
#     filterDoc=""
#     for text in doc.split('\n'):
#         filterDoc=filterDoc + stopWords(text)
#         filterDoc = filterDoc.rstrip(',')
#
#         filterDoc= filterDoc + "\n"
#
#     return filterDoc
#
# def stopWords(text):
#     filterText = ""
#     for token in gensim.utils.simple_preprocess(text):
#         if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
#             filterText = filterText + token
#             filterText = filterText + ","
#     return filterText

def vocab_size(data):
    vocab=set()
    for text in data:
        for word in text:
            vocab.add(word)

    return vocab,len(vocab)