from collections import Counter

import matplotlib.pyplot as plt
import nltk
from sklearn import datasets

from statistics import mean, stdev

news_dataset = datasets.fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
documents = news_dataset.data


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
    plt.savefig('boxplot_sentence_length.png')
    plt.show()


sentences_length, words = get_sentences_length_words(documents)
min_sentence_length = min(sentences_length)
max_sentence_length = max(sentences_length)
mean_sentence_length = mean(sentences_length)
std_sentence_length = stdev(sentences_length)
boxplot(sentences_length)


n_books = len(documents)
n_sentences = len(sentences_length)
n_words = len(words)
n_unique_words = len(Counter(words))

print('{0:>7} {1:>12} {2:>12} {3:>12}'.format('max', 'min', 'mean', 'std'))
print('{0:>7} {1:>12} {2:>12} {3:>12}'.format(max_sentence_length, min_sentence_length, int(mean_sentence_length), int(std_sentence_length)))
print('\n\n\n')
print('{0:>20} {1:>20} {2:>15} {3:>25} {4:>35}'.format('# of books', '# of sentences', '# of words', '# of unique words', 'mean # of words per sentence'))
print('{0:>20} {1:>20} {2:>15} {3:>25} {4:>35}'.format(n_books, n_sentences, n_words, n_unique_words, int(mean_sentence_length)))
