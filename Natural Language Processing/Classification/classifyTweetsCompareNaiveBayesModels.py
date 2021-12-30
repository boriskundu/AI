# Author: Boris Kundu
# Compare different Naive Bayes models for text classification using count and tfidf vectorizers.

#Import packages
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.feature_extraction.text as text
import sklearn.naive_bayes as nb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#Read data and create target class 'year'
df = pd.read_csv('tweets.csv')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year

#Function to run CountVectorizer
def runCountVetorizer(isBinary,X_train,Y_train,X_test,Y_test):
    countv = text.CountVectorizer(min_df = 10, ngram_range=(1,2), stop_words="english", binary=isBinary)
    X_train_tf = countv.fit_transform(X_train)
    X_test_tf = countv.transform(X_test)
    
    print(f'\n*** Running CountVectorizer with Binary={isBinary} ***')
    
    model1 = nb.BernoulliNB()
    model1.fit(X_train_tf, Y_train)
    Y_pred1 = model1.predict(X_test_tf)
    bernAcc = accuracy_score(Y_test, Y_pred1)
    print ('BernoulliNB Accuracy Score - ',bernAcc)
    
    model2 = nb.MultinomialNB()
    model2.fit(X_train_tf, Y_train)
    Y_pred2 = model2.predict(X_test_tf)
    multiAcc = accuracy_score(Y_test, Y_pred2)
    print ('MultinomialNB Accuracy Score - ', multiAcc)
    
    model3 = nb.ComplementNB()
    model3.fit(X_train_tf, Y_train)
    Y_pred3 = model3.predict(X_test_tf)
    compAcc = accuracy_score(Y_test, Y_pred3)
    print ('ComplementNB Accuracy Score - ', compAcc)
    
    return (bernAcc,multiAcc,compAcc)

#Function to run TfidfVectorizer
def runTfidfVectorizer(isBinary,X_train,Y_train,X_test,Y_test):
    tfidf = text.TfidfVectorizer(min_df = 10, ngram_range=(1,2), stop_words="english", binary=isBinary)
    X_train_tf = tfidf.fit_transform(X_train)
    X_test_tf = tfidf.transform(X_test)
    
    print(f'\n*** Running TfidfVectorizer with Binary={isBinary} ***')
    
    model1 = nb.BernoulliNB()
    model1.fit(X_train_tf, Y_train)
    Y_pred1 = model1.predict(X_test_tf)
    bernAcc = accuracy_score(Y_test, Y_pred1)
    print ('BernoulliNB Accuracy Score - ',bernAcc)
    
    model2 = nb.MultinomialNB()
    model2.fit(X_train_tf, Y_train)
    Y_pred2 = model2.predict(X_test_tf)
    multiAcc = accuracy_score(Y_test, Y_pred2)
    print ('MultinomialNB Accuracy Score - ', multiAcc)
    
    model3 = nb.ComplementNB()
    model3.fit(X_train_tf, Y_train)
    Y_pred3 = model3.predict(X_test_tf)
    compAcc = accuracy_score(Y_test, Y_pred3)
    print ('ComplementNB Accuracy Score - ', compAcc)
    
    return (bernAcc,multiAcc,compAcc)

#Function to run simulation
def splitRuns(df,epochs):
    bernAccBinL = []
    bernAccNoBinL = []
    bernAccBinTfL = []
    bernAccNoBinTfL = []
    
    multiAccBinL = []
    multiAccNoBinL = []
    multiAccBinTfL = []
    multiAccNoBinTfL = []
    
    compAccBinL = []
    compAccNoBinL = []
    compAccBinTfL = []
    compAccNoBinTfL = []
    
    for i in range(epoch):
        print(f'\n=== ITERATION {i+1} ===')
        X_train, X_test, Y_train, Y_test = train_test_split(df['text'],
                                                    df['year'],
                                                    test_size=0.3,
                                                    stratify=df['year'])
        print('Size of Training Data ', X_train.shape[0])
        print('Size of Test Data ', X_test.shape[0])
        
        #CountVectors with Binary
        (bernAccBin,multiAccBin,compAccBin) = runCountVetorizer(True,X_train,Y_train,X_test,Y_test)
        bernAccBinL.append(bernAccBin)
        multiAccBinL.append(multiAccBin)
        compAccBinL.append(compAccBin)
        #CountVectors without Binary
        (bernAccNoBin,multiAccNoBin,compAccNoBin) = runCountVetorizer(False,X_train,Y_train,X_test,Y_test)
        bernAccNoBinL.append(bernAccNoBin)
        multiAccNoBinL.append(multiAccNoBin)
        compAccNoBinL.append(compAccNoBin)
        #TfIdfVectors with Binary
        (bernAccBinTf,multiAccBinTf,compAccBinTf) = runTfidfVectorizer(True,X_train,Y_train,X_test,Y_test)
        bernAccBinTfL.append(bernAccBinTf)
        multiAccBinTfL.append(multiAccBinTf)
        compAccBinTfL.append(compAccBinTf)
        #TfIdfVectors without Binary
        (bernAccNoBinTf,multiAccNoBinTf,compAccNoBinTf) = runTfidfVectorizer(False,X_train,Y_train,X_test,Y_test)
        bernAccNoBinTfL.append(bernAccNoBinTf)
        multiAccNoBinTfL.append(multiAccNoBinTf)
        compAccNoBinTfL.append(compAccNoBinTf)
        
    fig, axes = plt.subplots(figsize=(8,6),num='Comparison')
    
    E = [i+1 for i in range(epoch)]
    
    axes.set_title('Model Comparison')
    axes.set_xlabel('Iteration')
    axes.set_ylabel('Accuracy')
    axes.plot(E,bernAccBinL,label='CountVectorizer BernoulliNB Binary',marker='o')
    axes.plot(E,multiAccBinL,label='CountVectorizer MultinomialNB Binary',marker='o')
    axes.plot(E,compAccBinL,label='CountVectorizer ComplementNB Binary',marker='o')
    axes.plot(E,bernAccNoBinL,label='CountVectorizer BernoulliNB Non-Binary',marker='o')
    axes.plot(E,multiAccNoBinL,label='CountVectorizer MultinomialNB Non-Binary',marker='o')
    axes.plot(E,compAccNoBinL,label='CountVectorizer ComplementNB Non-Binary',marker='o')                    
    axes.plot(E,bernAccBinTfL,label='TfidfVectorizer BernoulliNB Binary',marker='o')
    axes.plot(E,multiAccBinTfL,label='TfidfVectorizer MultinomialNB Binary',marker='o')
    axes.plot(E,compAccBinTfL,label='TfidfVectorizer ComplementNB Binary',marker='o')
    axes.plot(E,bernAccNoBinTfL,label='TfidfVectorizer BernoulliNB Non-Binary',marker='o')
    axes.plot(E,multiAccNoBinTfL,label='TfidfVectorizer MultinomialNB Non-Binary',marker='o')
    axes.plot(E,compAccNoBinTfL,label='TfidfVectorizer ComplementNB Non-Binary',marker='o')
    axes.legend()
    
    plt.tight_layout()
    plt.show()

#Run simulation
epoch = 5
splitRuns(df,epoch)