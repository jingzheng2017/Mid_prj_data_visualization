import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from gensim.parsing.porter import PorterStemmer
import pyLDAvis.gensim
from gensim import corpora
from gensim.models import LdaModel
from collections import defaultdict
from sklearn import datasets


def preprocessing_document(document):
    # lower the words
    document = document.lower()
    # remove special symbols
    document = re.sub('[^ a-z]+', '', document)

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


def get_corpus_dictionary(texts):

    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > 1]
             for text in texts]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    return corpus, dictionary


def test_lda(texts):
    corpus, dictionary = get_corpus_dictionary(texts)
    lda = LdaModel(corpus=corpus, num_topics=20)
    data = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    pyLDAvis.show(data, open_browser=False)


news_dataset = datasets.fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
documents = news_dataset.data
texts = []
for i in range(len(documents)):
    texts.append(preprocessing_document(documents[i]))

if __name__ == "__main__":
    test_lda(texts)

