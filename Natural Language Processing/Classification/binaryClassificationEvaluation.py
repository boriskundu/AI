# Author:Boris Kundu
# Binary Classification Evaluation
# Mutual info and Chi square ranking of features for years in tweets with 2-classes (before and after 2016)

#Import packages
import pandas as pd
import numpy as np
import sklearn.naive_bayes as nb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.feature_selection as fs
from sklearn.metrics import accuracy_score
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
import matplotlib.pyplot as plt

#Add stop words
stopwords.add('amp')
stopwords.add('https')
stopwords.add('http')
stopwords.add('co')
stopwords.add('rt')

#Read data and get target binary class year (one before 2016 and other after 2016)
df = pd.read_csv('tweets.csv')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year >= 2016
#Split data
X_train, X_test, Y_train, Y_test = train_test_split(df['text'],
                                                    df['year'],
                                                    test_size=0.3,
                                                    stratify=df['year'])
#Using CountVectorizer
biv = CountVectorizer(min_df = 10, ngram_range=(1,2), stop_words=stopwords, binary=True)
X_train_bi = biv.fit_transform(X_train)
X_test_bi = biv.transform(X_test)
#Train and fit BernoulliNB
model1 = nb.BernoulliNB()
model1.fit(X_train_bi, Y_train)
#Predict BernoulliNB
Y_pred = model1.predict(X_test_bi)
print ('Original BernoulliNB Accuracy Score - ', accuracy_score(Y_test, Y_pred))
#Train and fit MultinomialNB
model1 = nb.MultinomialNB()
model1.fit(X_train_bi, Y_train)
#Predict MultinomialNB
Y_pred = model1.predict(X_test_bi)
print ('Original MultinomialNB Accuracy Score - ', accuracy_score(Y_test, Y_pred))
#Feature selection size for score calculation
selection_sizes = [200,400,600,800,1000,1200,1400,1600]

#Function to evaluate
def getBestSelectionSize(selection_sizes):
	mi_score_b = []
	mi_score_m = []
	chi_score_b = []
	chi_score_m = []

	for size in selection_sizes:
		print(f'\nRunning Feature Selection using Mutual Info with Selection Size:{size}')
		#Mi
		selector_mi = fs.SelectKBest(fs.mutual_info_classif, k=size)
		X_train_mi = selector_mi.fit_transform(X_train_bi, Y_train)
		X_test_mi = selector_mi.transform(X_test_bi)

		model2 = nb.BernoulliNB()
		model2.fit(X_train_mi, Y_train)

		Y_pred = model2.predict(X_test_mi)
		accuracy = accuracy_score(Y_test, Y_pred)
		mi_score_b.append(accuracy)
		print (f'Mutual Info with Selection Size:{size} for BernoulliNB has Accuracy Score:{accuracy}')

		model2 = nb.MultinomialNB()
		model2.fit(X_train_mi, Y_train)

		Y_pred = model2.predict(X_test_mi)
		accuracy = accuracy_score(Y_test, Y_pred)
		mi_score_m.append(accuracy)
		print (f'Mutual Info with Selection Size:{size} for MultinomialNB has Accuracy Score:{accuracy}')

		print(f'\nRunning Feature Selection using Chi Square with Selection Size:{size}')
		#Chi
		selector_chi = fs.SelectKBest(fs.chi2, k=size)
		X_train_chi = selector_chi.fit_transform(X_train_bi, Y_train)
		X_test_chi = selector_chi.transform(X_test_bi)

		model2 = nb.BernoulliNB()
		model2.fit(X_train_chi, Y_train)

		Y_pred = model2.predict(X_test_chi)
		accuracy = accuracy_score(Y_test, Y_pred)
		chi_score_b.append(accuracy)
		print (f'Chi Square with Selection Size:{size} for BernoulliNB has Accuracy Score:{accuracy}')

		model2 = nb.MultinomialNB()
		model2.fit(X_train_chi, Y_train)

		Y_pred = model2.predict(X_test_chi)
		accuracy = accuracy_score(Y_test, Y_pred)
		chi_score_m.append(accuracy)
		print (f'Chi Square with Selection Size:{size} for MultinomialNB has Accuracy Score:{accuracy}')

	return (mi_score_b,mi_score_m,chi_score_b,chi_score_m)

(mi_score_b,mi_score_m,chi_score_b,chi_score_m) = getBestSelectionSize(selection_sizes)

#Function to plot and compare
def compareModels(selection_sizes,mi_score_b,mi_score_m,chi_score_b,chi_score_m):
	fig, axes = plt.subplots(figsize=(8,6),num='Comparison')

	axes.set_title('Comparison')
	axes.set_xlabel('Selection Size')
	axes.set_ylabel('Accuracy')
	axes.plot(selection_sizes,mi_score_b,label='BernoulliNB Mutual Info',marker='o')
	axes.plot(selection_sizes,mi_score_m,label='MultinomialNB Mutual Info',marker='o')
	axes.plot(selection_sizes,chi_score_b,label='BernoulliNB Chi Square',marker='o')
	axes.plot(selection_sizes,chi_score_m,label='MultinomialNB Chi Square',marker='o')

	axes.legend()

	plt.tight_layout()
	plt.show()

#Compare models
compareModels(selection_sizes,mi_score_b,mi_score_m,chi_score_b,chi_score_m)