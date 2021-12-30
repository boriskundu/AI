# Author: BORIS KUNDU
# Comparison of different tokenizers on tweets.csv

#Import packages
import pandas as pd
import numpy as np
import regex as re
import nltk

#Setup for RegularExpression
RE_TOKEN = re.compile(r"""
               ( [#]?[@\w'’\.\-\:]*\w     # words, hash tags and email adresses
               | [:;<]\-?[\)\(3]          # coarse pattern for basic text emojis
               | [\U0001F100-\U0001FFFF]  # coarse code range for unicode emojis
               )
               """, re.VERBOSE)

#Read tweets.csv
df = pd.read_csv('tweets.csv')

#Function to tokenize and display
def displayTweetsWithTokens (numOfTweets):
    print(f'Displaying top {numOfTweets} Tweets with Tokens from different Tokenizers.')
    for i in range (numOfTweets):
        #Display Tweet
        print(f'*** TWEET {i+1} *** => {df["text"][i]}')
        #Display TweetTokenizer result
        tweetTokenizer = nltk.tokenize.TweetTokenizer()
        tokens = tweetTokenizer.tokenize(df['text'][i])
        print('*** TweetTokenizer Tokens ***')
        print(*tokens, sep='|')
        #Display WordTokenizer result
        tokens = nltk.tokenize.word_tokenize(df['text'][i])
        print('*** WordTokenizer Tokens ***')
        print(*tokens, sep='|')
        #Display RegularExpressionTokenizer
        regularExpressionTokenizer = nltk.tokenize.RegexpTokenizer(RE_TOKEN.pattern, flags=re.VERBOSE)
        tokens = regularExpressionTokenizer.tokenize(df['text'][i])
        print('*** RegularExpressionTokenizer Tokens ***')
        print(*tokens, sep='|')
        #Display ToktokTokenizer
        toktokTokenizer = nltk.tokenize.ToktokTokenizer()
        tokens = toktokTokenizer.tokenize(df['text'][i])
        print('*** ToktokTokenizer Tokens ***')
        print(*tokens, sep='|')
        print('\n')

#Take user input
numOfTweets = input('Enter the number of tweets you want to tokenize differently:')
numTweets = int(numOfTweets)
#Display results
displayTweetsWithTokens(numTweets)