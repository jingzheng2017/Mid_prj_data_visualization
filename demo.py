from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
import os


def preprocessing_document(document):
    # lower the words
    document = document.lower()
    # remove special symbols
    document = re.sub('[^ a-z0-9]+', '', document)

    # remove punctuation
    tokenizer = RegexpTokenizer(r'\w+')

    # tokenize document
    tokenizer.tokenize(document)

    # remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(document)

    document_words = [w for w in word_tokens if w not in stop_words]

    ps = PorterStemmer()
    wnl = WordNetLemmatizer()
    for i in range(len(document_words)):
        document_words[i] = wnl.lemmatize(ps.stem(document_words[i]))

    return document_words


example_sent = "Eighty-seven &*#miles game gaming saw games to go, yet.  Onward!"
example_sent = 'he saw many ghosts'
result = preprocessing_document(example_sent)

