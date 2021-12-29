# Author: Boris Kundu
# Plot trend by year for any input word coming in 50 tweets.

#Import packages
import pandas as pd
import sys
import matplotlib.pyplot as plt

#Read data
df = pd.read_csv('tweets.csv')

#Extraxt year from date
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year

#Get word
word = input("Enter a word to search in Trump's tweets and plot trend by year:")

#Get matching tweets
if len(sys.argv) > 1:
    word = sys.argv[1]
subset = df[df['text'].str.contains(word)]

#Plot trend if tweets > 50
if len(subset) > 50:
    subset.groupby('year').size().plot(title=word)
    plt.xlabel('Year')
    plt.ylabel('Tweets')
    plt.show()
