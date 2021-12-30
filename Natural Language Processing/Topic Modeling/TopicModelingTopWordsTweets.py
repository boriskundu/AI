# Author: BORIS KUNDU
# Topic Modeling on tweets using Non-ngeative Matrix Factorization.
# Gets top words(features) and top tweets per topic and their word clouds.

#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
from sklearn.decomposition import NMF
from wordcloud import WordCloud

#Read data
df = pd.read_csv('tweets.csv')

#Get tfidf vectorized data
tfidf_text_vectorizer = TfidfVectorizer(stop_words=stopwords, min_df=5, max_df=0.7)
tfidf_text_vectors = tfidf_text_vectorizer.fit_transform(df['text'])

#Use NMF for getting 5 topics
nmf_model = NMF(n_components=5)
W_tweet_matrix = nmf_model.fit_transform(tfidf_text_vectors)
H_tweet_matrix = nmf_model.components_

#Get features aka tokens aka terms
features = tfidf_text_vectorizer.get_feature_names()

#Get top tweets
def getTweetRank(tweets,wordlist,scoreList):
    tweetScore = []
    for t in range(len(tweets)):
        score = 0
        n = len(wordlist)
        for i in range(n):
            weight = scoreList[i]
            word = wordlist[i]
            score = score + tweets[t].lower().count(word.lower())*weight
        tweetScore.append(score)
    return tweetScore

#Display top tweets per topic
def display_topics(model, components, features, no_top_words=5):
    for topic, word_vector in enumerate(components):
        print(f'\n*** Topic:{topic+1} ***')
        top_scores = []
        top_words = []
        tweetRank = []
        total = word_vector.sum()
        largest = word_vector.argsort()[::-1] # invert sort order
        print(f'\nTop Words:\n')
        for i in range(no_top_words):
            word = features[largest[i]]
            score = word_vector[largest[i]]*100.0/total
            print(" %s (%2.2f)" % (word,score))
            top_words.append(word)
            top_scores.append(score)
        tweetRank = getTweetRank(df['text'],top_words,top_scores)
        print('\nTop Tweets(ordered):')
        indexes = np.argsort(tweetRank)[-5:]
        for i in indexes:
            print('\n'+df['text'][i])

#Display word cloud
def wordcloud_topics(model, components, features, no_top_words=40):
    for topic, words in enumerate(components):
        size = {}
        largest = words.argsort()[::-1] # invert sort order
        for i in range(0, no_top_words):
            size[features[largest[i]]] = abs(words[largest[i]])
        wc = WordCloud(background_color="white", max_words=100, width=960, height=540)
        wc.generate_from_frequencies(size)
        plt.title(f'Topic {topic+1} Word Cloud')
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.show()

display_topics(W_tweet_matrix,H_tweet_matrix,features)

wordcloud_topics(W_tweet_matrix,H_tweet_matrix,features)