from collections import Counter

import matplotlib.pyplot as plt
import nltk
import re
from gensim.parsing.porter import PorterStemmer
from gensim.parsing.preprocessing import remove_stopwords
from sklearn import datasets
from statistics import mean, stdev


def preprocessing_documents(documents):
    for i in range(len(documents)):
        documents[i] = documents[i].lower()
        documents[i] = re.sub('[^ a-z\.\?\!]+', '', documents[i])
        documents[i] = remove_stopwords(documents[i])

    return documents


def get_sentences_length_words(documents):
    sentences_length = []
    words = []
    for i in range(len(documents)):
        sentences = nltk.sent_tokenize(documents[i])
        for j in range(len(sentences)):
            words_in_sentence = sentences[j].split()
            sentences_length.append(len(words_in_sentence))
            for k in range(len(words_in_sentence)):
                words.append(words_in_sentence[k])

    return sentences_length, words


def boxplot(sentence_length):
    plt.boxplot(sentences_length, labels=['Sentence Length of Corpus'])
    plt.savefig('./results/boxplot_sentence_length.png')
    plt.show()


def plot_tf(words):
    freqdist = nltk.FreqDist(words)
    plt.figure(figsize=(16, 5))
    freqdist.plot(50)


def postprocess_words(words):
    for i in range(len(words)):
        p = PorterStemmer()
        words[i] = p.stem(words[i])
    return words


if __name__ == "__main__":
    news_dataset = datasets.fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    documents = preprocessing_documents(news_dataset.data)

    sentences_length, words = get_sentences_length_words(documents)
    min_sentence_length = min(sentences_length)
    max_sentence_length = max(sentences_length)
    mean_sentence_length = mean(sentences_length)
    std_sentence_length = stdev(sentences_length)
    boxplot(sentences_length)


    n_books = len(documents)
    n_sentences = len(sentences_length)
    n_words = len(words)
    words = postprocess_words(words)
    n_unique_words = len(Counter(words))

    print('{0:>7} {1:>12} {2:>12} {3:>12}'.format('max', 'min', 'mean', 'std'))
    print('{0:>7} {1:>12} {2:>12} {3:>12}'.format(max_sentence_length, min_sentence_length, int(mean_sentence_length), int(std_sentence_length)))
    print('\n\n')
    print('{0:>7} {1:>20} {2:>15} {3:>25} {4:>35}'.format('# of books', '# of sentences', '# of words', '# of unique words', 'mean # of words per sentence'))
    print('{0:>7} {1:>20} {2:>15} {3:>25} {4:>35}'.format(n_books, n_sentences, n_words, n_unique_words, int(mean_sentence_length)))

    plot_tf(words)