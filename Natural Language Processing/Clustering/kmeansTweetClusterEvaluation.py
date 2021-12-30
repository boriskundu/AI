# Author: Boris Kundu
# K-Means clustering of tweets with evaluated against year

#Import packages
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
from sklearn.cluster import KMeans
import sklearn.metrics.cluster as cluster
import sklearn.metrics as metrics
#Read data and get year from date
df = pd.read_csv('tweets.csv')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
print(f'Tweets by Year:\n{df.groupby("year").size()}')
#Add more stop words
stopwords.add('amp')
stopwords.add('https')
stopwords.add('http')
stopwords.add('co')
stopwords.add('rt')

#Get TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords, min_df=5, max_df=0.7)
tfidf_vectors = tfidf_vectorizer.fit_transform(df["text"])
print(f'Shape:\n{tfidf_vectors.shape}')
years = df['year'].tolist()

K = 10
k_means = KMeans(n_clusters=K)
k_means.fit(tfidf_vectors)
sizes = []
print(f'\n*** KMeans with K={K} START ***')
for i in range(K):
    sizes.append({"cluster": i, "size": np.sum(k_means.labels_==i)})
print(sizes)
cm = cluster.contingency_matrix(years, k_means.labels_)
fms = metrics.fowlkes_mallows_score(years, k_means.labels_)
rs = metrics.rand_score(years, k_means.labels_)
amis = metrics.adjusted_mutual_info_score(years, k_means.labels_)
mis = metrics.mutual_info_score(years, k_means.labels_)
nmis = metrics.normalized_mutual_info_score(years, k_means.labels_)

print(f'\n Contingency Matrix:\n{cm}')
print(f'\n Fowlkes Mallows Score:\n{fms}')
print(f'\n Rand Score:\n{rs}')
print(f'\n Adjusted Mutual Info Score:\n{amis}')
print(f'\n Mutual Info Score:\n{mis}')
print(f'\n Normalized Mutual Info Score:\n{nmis}')

print(f'*** KMeans with K={K} END ***')

K = 8
k_means = KMeans(n_clusters=K)
k_means.fit(tfidf_vectors)
sizes = []
print(f'\n*** KMeans with K={K} START ***')
for i in range(K):
    sizes.append({"cluster": i, "size": np.sum(k_means.labels_==i)})
print(sizes)
cm = cluster.contingency_matrix(years, k_means.labels_)
fms = metrics.fowlkes_mallows_score(years, k_means.labels_)
rs = metrics.rand_score(years, k_means.labels_)
amis = metrics.adjusted_mutual_info_score(years, k_means.labels_)
mis = metrics.mutual_info_score(years, k_means.labels_)
nmis = metrics.normalized_mutual_info_score(years, k_means.labels_)

print(f'\n Contingency Matrix:\n{cm}')
print(f'\n Fowlkes Mallows Score:\n{fms}')
print(f'\n Rand Score:\n{rs}')
print(f'\n Adjusted Mutual Info Score:\n{amis}')
print(f'\n Mutual Info Score:\n{mis}')
print(f'\n Normalized Mutual Info Score:\n{nmis}')

print(f'*** KMeans with K={K} END ***')

K = 12
k_means = KMeans(n_clusters=K)
k_means.fit(tfidf_vectors)
sizes = []
print(f'\n*** KMeans with K={K} START ***')
for i in range(K):
    sizes.append({"cluster": i, "size": np.sum(k_means.labels_==i)})
print(sizes)
cm = cluster.contingency_matrix(years, k_means.labels_)
fms = metrics.fowlkes_mallows_score(years, k_means.labels_)
rs = metrics.rand_score(years, k_means.labels_)
amis = metrics.adjusted_mutual_info_score(years, k_means.labels_)
mis = metrics.mutual_info_score(years, k_means.labels_)
nmis = metrics.normalized_mutual_info_score(years, k_means.labels_)

print(f'\n Contingency Matrix:\n{cm}')
print(f'\n Fowlkes Mallows Score:\n{fms}')
print(f'\n Rand Score:\n{rs}')
print(f'\n Adjusted Mutual Info Score:\n{amis}')
print(f'\n Mutual Info Score:\n{mis}')
print(f'\n Normalized Mutual Info Score:\n{nmis}')

print(f'*** KMeans with K={K} END ***')