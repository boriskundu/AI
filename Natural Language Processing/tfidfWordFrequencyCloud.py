# Author: Boris Kundu
# Word Frequency Analysis using tfidf word ranking on tweets.csv

#Import packages
import pandas as pd
import numpy as np
import regex as re
import nltk
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
from matplotlib import pyplot as plt
#Get stop words
stopwords = set(nltk.corpus.stopwords.words('english'))
#Read data
df = pd.read_csv("tweets.csv")
#Tokenize using regular expression
def tokenize(text):
	return re.findall(r'[\w-]*\p{L}[\w-]*', text)
#Remove stop words
def remove_stop(tokens):
	return [t for t in tokens if t not in stopwords]
#Define pre-processing pipeline
pipeline = [str.lower, tokenize, remove_stop]
#Prepare data
def prepare(text, pipeline):
	tokens = text
	for transform in pipeline:
		tokens = transform(tokens)
	return tokens
#Get tokens using pre-processing pipeline
df['tokens'] = df['text'].apply(prepare, pipeline=pipeline)
#Get year from date
df['year']=(pd.to_datetime(df['date'])).dt.year
#Define and update counter for word frequency
counter = Counter()
df['tokens'].map(lambda x: counter.update(set(x)))
#Number of rows = total documents
N = len(df)
#Calculate inverse document frequency
idf = dict(map(lambda x: (x[0], np.log(N/x[1]) + 0.1), counter.items()))
#For tweets from 2015 to 2019
for y in range(2015, 2019):
	#Clean
	counter.clear()
	#Yearly tokens
	df[df['year'] == y]['tokens'].map(counter.update)
	#Calculcate tfidf = term frequency * idf
	tfidf = dict(map(lambda x: (x[0], x[1] * idf.get(x[0])), counter.items()))
	#Define word cloud
	wc = WordCloud()
	wc.generate_from_frequencies(tfidf)
	#Plot word cloud by year
	plt.imshow(wc, interpolation='bilinear')
	plt.title(str(y))
	plt.axis("off")
	plt.show()