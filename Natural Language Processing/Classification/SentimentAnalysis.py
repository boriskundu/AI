# Author: Boris Kundu
# Sentiment analyzer using Lexicon Score and Linear SVC 

#Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import html
import re
import nltk
import textacy.preprocessing as tprep
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from nltk.corpus import opinion_lexicon
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import scale

#Read data
df = pd.read_json('Luxury_Beauty.json',lines=True)

#Check sample
print(f':Check sample data:\n{df.sample(1).T}')

#Check for missing data
print(f':Check missing data:\n{df.isna().sum()}')

# Assigning a new [1,0] target class label based on the product rating
df['sentiment'] = 0
df.loc[df['overall'] > 3, 'sentiment'] = 1
df.loc[df['overall'] < 3, 'sentiment'] = 0

# Removing unnecessary columns to keep a simple DataFrame
df.drop(columns=['reviewTime', 'unixReviewTime', 'reviewerID', 'summary','reviewerName','vote','image','style'],axis=1,inplace=True)
df.sample(3)

#Normalize data
def normalize(text):
    text = tprep.normalize.hyphenated_words(text)
    text = tprep.normalize.quotation_marks(text)
    text = tprep.normalize.unicode(text)
    text = tprep.remove.accents(text)
    return text

#Function to clean text data
def clean(text):
    # convert html escapes like &amp; to characters.
    text = html.unescape(text)
    # tags like <tab>
    text = re.sub(r'<[^<>]*>', ' ', text)
    # markdown URLs like [Some text](https://....)
    text = re.sub(r'\[([^\[\]]*)\]\([^\(\)]*\)', r'\1', text)
    # text or code in brackets like [0]
    text = re.sub(r'\[[^\[\]]*\]', ' ', text)
    # standalone sequences of specials, matches &# but not #cool
    text = re.sub(r'(?:^|\s)[&#<>{}\[\]+|\\:-]{1,}(?:\s|$)', ' ', text)
    # standalone sequences of hyphens like --- or ==
    text = re.sub(r'(?:^|\s)[\-=\+]{2,}(?:\s|$)', ' ', text)
    # sequences of white spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

#Check for umpurtity. Lower is better
RE_SUSPICIOUS = re.compile(r'[&#<>{}\[\]\\]')

#Returns share of suspicious characters
def impurity(text, min_len=4):
    if text == None or len(text) < min_len:
        return 0
    else:
        return len(RE_SUSPICIOUS.findall(text))/len(text)

#Yake backup
df['review_orig'] = df['reviewText'].copy()
#Convert to string
df['reviewText'] = df['reviewText'].apply(str)
#Conver to lower case
df['reviewText'] = df['reviewText'].apply(str.lower)
# add new column to data frame
df['impurity'] = df['reviewText'].apply(impurity, min_len=4)
# get the top 3 records
df[['reviewText', 'impurity']].sort_values(by='impurity', ascending=False).head(3)
#Normalize and Clean data
df['reviewText'] = df['reviewText'].apply(normalize)
df['reviewText'] = df['reviewText'].apply(clean)
# Remove observations that are empty after the cleaning step
df = df[df['reviewText'].str.len() != 0]

#Downlaod opinion lexicon
#nltk.download('opinion_lexicon')

#Check positive and negative words
print('Total number of words in opinion lexicon', len(opinion_lexicon.words()))
print('Examples of positive words in opinion lexicon',opinion_lexicon.positive()[:5])
print('Examples of negative words in opinion lexicon',opinion_lexicon.negative()[:5])

# Positive and Negative words
df.rename(columns={"reviewText": "text"}, inplace=True)
pos_score = 1
neg_score = -1
word_dict = {}
# Adding the positive words to the dictionary
for word in opinion_lexicon.positive():
    word_dict[word] = pos_score
# Adding the negative words to the dictionary
for word in opinion_lexicon.negative():
    word_dict[word] = neg_score

#Lexicon scorer
def lexicon_scorer(text):
    sentiment_score = 0
    bag_of_words = word_tokenize(text.lower())
    for word in bag_of_words:
        if word in word_dict:
            sentiment_score += word_dict[word]
    return sentiment_score / len(bag_of_words)

#Get lexicon score
df['lexicon_score'] = df['text'].apply(lexicon_scorer)

#Scale score
df['lexicon_score'] = scale(df['lexicon_score'])

#Check sample
print(f"Samples:\n{df[['text','lexicon_score']].sample(2)}")

#Check mean
print(f"Mean lexicon score by rating:{df.groupby('overall')['lexicon_score'].mean()}")

#Lexicon scorer
def baseline_scorer(text):
    score = lexicon_scorer(text)
    if score > -0.035: #Mean score for tarting 3
        return 1
    else:
        return 0

#Split data into train and test sets. 
X_train, X_test, Y_train, Y_test = train_test_split(df['text'],
                                                    df['sentiment'],
                                                    test_size=0.3,
                                                    random_state=101,
                                                    stratify=df['sentiment'])

print ('Size of Training Data ', X_train.shape[0])
print ('Size of Test Data ', X_test.shape[0])
print ('Distribution of classes in Training Data :')
print ('Positive Sentiment ', str(sum(Y_train == 1)/ len(Y_train) * 100.0))
print ('Negative Sentiment ', str(sum(Y_train == 0)/ len(Y_train) * 100.0))
print ('Distribution of classes in Testing Data :')
print ('Positive Sentiment ', str(sum(Y_test == 1)/ len(Y_test) * 100.0))
print ('Negative Sentiment ', str(sum(Y_test == 0)/ len(Y_test) * 100.0))

#Make predictions 
Y_pred_baseline = X_test.apply(baseline_scorer)
acc_score = accuracy_score(Y_pred_baseline, Y_test)
print (f'Lexicon Sentiment Analyzer Accuracy:{acc_score}')

#text Vectorizer
tfidf = TfidfVectorizer(min_df = 10, ngram_range=(1,1))
X_train_tf = tfidf.fit_transform(X_train)
X_test_tf = tfidf.transform(X_test)

#Train model
model1 = LinearSVC(random_state=42, tol=1e-5)
model1.fit(X_train_tf, Y_train)

#Predict and get accuracy
Y_pred = model1.predict(X_test_tf)
print ('Linear SVC Accuracy- ',accuracy_score(Y_test, Y_pred))
print ('Linear SVC ROC-AUC Score - ',roc_auc_score(Y_test, Y_pred))

#!pip install transformers
#from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
#config = BertConfig.from_pretrained('bert-base-uncased',finetuning_task='binary')