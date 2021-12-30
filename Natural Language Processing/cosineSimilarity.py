# Author: Boris Kundu
# Finding top 5 most similar headlines to a randomly selected headline.

#Import packages
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Read data
documents = pd.read_csv('abcnews-date-text.csv', parse_dates=["publish_date"])
#Get tfidf
tfidf = TfidfVectorizer()
dt = tfidf.fit_transform(documents["headline_text"])
#Get random headline
query = random.randrange(len(documents))
print(f'Randomly selected headline:{documents.iloc[query]["headline_text"]}')
#Replacing string with the randomly generated one from IR9G_BORIS_KUNDU.py
made_up = tfidf.transform(documents.iloc[query][["headline_text"]])
#Apply cosine similarity
sim = cosine_similarity(made_up, dt)
print(f'Top 5 matching headlines are mentioned below:')
print(documents.iloc[np.argsort(sim[0])[::-1][0:5]][["publish_date", "headline_text"]])