# Author: Boris Kundu
# TruncatedSVD and LatentDirichletAllocation for the term document matrix from tweets

#Import packages
import pandas as pd
import matplotlib.pyplot as plt
import re
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud

#Read data
df = pd.read_csv('tweets.csv')
#CountVectorizer
count_tweet_vectorizer = CountVectorizer(stop_words=stopwords, min_df=5, max_df=0.7)
count_text_vectors = count_tweet_vectorizer.fit_transform(df["text"])
#TfidfVectorizer
tfidf_text_vectorizer = TfidfVectorizer(stop_words=stopwords, min_df=5, max_df=0.7)
tfidf_text_vectors = tfidf_text_vectorizer.fit_transform(df['text'])
#Function to display top 5 words
def display_topics(title, model, features, no_top_words=5):
    for topic, word_vector in enumerate(model.components_):
        total = word_vector.sum()
        largest = word_vector.argsort()[::-1] # invert sort order
        print(f'\n{title} Topic {topic + 1}')
        for i in range(0, no_top_words):
            print(" %s (%2.2f)" % (features[largest[i]],
                   word_vector[largest[i]]*100.0/total))

#Function to display word cloud for top 10 words
def wordcloud_topics(title,model, features, no_top_words=10):
    for topic, words in enumerate(model.components_):
        size = {}
        largest = words.argsort()[::-1] # invert sort order
        for i in range(0, no_top_words):
            size[features[largest[i]]] = abs(words[largest[i]])
        wc = WordCloud(background_color="white", max_words=100, width=960, height=540)
        wc.generate_from_frequencies(size)
        plt.title(f'{title} Topic {topic+1}')
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.show()

#LatentDirichletAllocation - 3 topics
lda_tweet_model = LatentDirichletAllocation(n_components = 3)
W_lda_para_matrix = lda_tweet_model.fit_transform(count_text_vectors)
H_lda_para_matrix = lda_tweet_model.components_
#TruncatedSVD - 3 topics
svd_para_model = TruncatedSVD(n_components = 3)
W_svd_para_matrix = svd_para_model.fit_transform(tfidf_text_vectors)
H_svd_para_matrix = svd_para_model.components_

display_topics('SVD',svd_para_model, tfidf_text_vectorizer.get_feature_names())
display_topics('LDA',lda_tweet_model, count_tweet_vectorizer.get_feature_names())

wordcloud_topics('SVD',svd_para_model, tfidf_text_vectorizer.get_feature_names())
wordcloud_topics('LDA',lda_tweet_model, count_tweet_vectorizer.get_feature_names())