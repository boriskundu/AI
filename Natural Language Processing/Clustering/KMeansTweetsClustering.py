# Author: Boris Kundu
# K-Means clustering of tweets data

#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
from sklearn.cluster import KMeans

# Read data
df = pd.read_csv('tweets.csv')
#Create and transform using TfidfVectorizer 
tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords, min_df=5, max_df=0.7)
tfidf_vectors = tfidf_vectorizer.fit_transform(df["text"])

#Train and fit K-Means
k_means_para = KMeans(n_clusters=5)
k_means_para.fit(tfidf_vectors)

#Display top 30 words aka features in each cluster
def wordcloud_clusters(model, vectors, features, no_top_words=30):
    for cluster in np.unique(model.labels_):
        size = {}
        words = vectors[model.labels_ == cluster].sum(axis=0).A[0]
        largest = words.argsort()[::-1] # invert sort order
        for i in range(0, no_top_words):
            size[features[largest[i]]] = abs(words[largest[i]])
        wc = WordCloud(background_color="white", max_words=100, width=960, height=540)
        wc.generate_from_frequencies(size)
        plt.title(f'Cluster {cluster+1} Top 30 Words')
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.show()

#Find cluster sizes
sizes = []
for i in range(5):
    sizes.append({"cluster": i+1, "size": np.sum(k_means_para.labels_==i)})

#Plot cluster sizes
pd.DataFrame(sizes).set_index("cluster").plot.bar(figsize=(16,9))
plt.title('Cluster Size Bar Plot')
plt.show()

print(f'Cluster Labels:{np.unique(k_means_para.labels_, return_counts=True)}')

wordcloud_clusters(k_means_para, tfidf_vectors, tfidf_vectorizer.get_feature_names())