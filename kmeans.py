# from sklearn.datasets import fetch_20newsgroups
# from sklearn.feature_extraction.text import TfidfVectorizer
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn import metrics
#
#
# newsgroups_train = fetch_20newsgroups(subset='train')
# labels = newsgroups_train.target
#
# true_k = 20
#
# #vectorize
#
# vectorizer = TfidfVectorizer(max_df=0.5,
#                              min_df=2,
#                              stop_words='english')
#
#
# X = vectorizer.fit_transform(newsgroups_train.data)
#
# #clustering
# km = KMeans(n_clusters=20, init='k-means++', max_iter=100, n_init=1)
#
# km.fit(X)
# y_kmeans = km.predict(X)
#
# order_centroids = km.cluster_centers_.argsort()[:, ::-1]
#
# terms = vectorizer.get_feature_names()
#
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


newsgroups_train = fetch_20newsgroups(subset='train', categories=['alt.atheism', 'sci.space', 'talk.politics.misc'])
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
])
X = pipeline.fit_transform(newsgroups_train.data).todense()

pca = PCA(n_components=2).fit(X)
data2D = pca.transform(X)
labels = newsgroups_train.target

kmeans = KMeans(n_clusters=3).fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(data2D[:,0], data2D[:,1], c=y_kmeans)
plt.savefig('./results/kmeans.png')
plt.show()
