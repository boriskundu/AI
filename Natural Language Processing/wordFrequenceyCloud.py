# Author: Borius Kundu 
# Word Frequency Analysis on un-general-debates.csv

#Import packages
import pandas as pd
import regex as re
import nltk
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
from matplotlib import pyplot as plt

#Get stop words
stopwords = set(nltk.corpus.stopwords.words('english'))
#Read data
df = pd.read_csv("un-general-debates.csv")
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
#Define and update counter for word frequency
counter = Counter()
df['tokens'].map(counter.update)
#Define word cloud
wc = WordCloud()
wc.generate_from_frequencies(counter)
#Plot word cloud
plt.title('Word Cloud using Word Frequency')
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()