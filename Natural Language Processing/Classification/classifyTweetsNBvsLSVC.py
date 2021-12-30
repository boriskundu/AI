# Author: BORIS KUNDU
# Compare Multinomial Naive Bayes with Linear SVC using Count Vectorizer on tweets data.
# Used different minimum document ferequency and ngram ranges.

# Import packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix, classification_report  
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

#Read data and create target class 'year'
df = pd.read_csv('tweets.csv')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year

#Split data
X_train, X_test, Y_train, Y_test = train_test_split(df['text'],
                                                    df['year'],
                                                    test_size = 0.3,
                                                    stratify = df['year'])
#Display size of dataset
print('Size of Training Data ', X_train.shape[0])
print('Size of Test Data ', X_test.shape[0])


#Function to train, predict and compare models 
def compareNBwithSVC(mindf,ngramrange):
    title = 'min_df:'+str(mindf)+' and ngram_range:'+str(ngramrange)
    print('Running training simulation with '+title)
    countv = CountVectorizer(min_df = mindf, ngram_range=ngramrange, stop_words="english")
    X_train_tf = countv.fit_transform(X_train)
    X_test_tf = countv.transform(X_test)
    #Naive Bayes
    model1 = MultinomialNB()
    model1.fit(X_train_tf, Y_train)
    #LinearSVC
    model2 = LinearSVC()
    model2.fit(X_train_tf, Y_train)
    #Predictions
    Y_pred_nb = model1.predict(X_test_tf)
    Y_pred_svc = model2.predict(X_test_tf)
    #Accuracy                           
    print(f'\nNaive Bayes Accuracy Score - {accuracy_score(Y_test, Y_pred_nb)}')
    print(f'Linear SVC Accuracy Score - {accuracy_score(Y_test, Y_pred_svc)}')
    #Confusion Matrix
    print('\nNaive Bayes Confusion Matrix')
    print(confusion_matrix(Y_test, Y_pred_nb))
    print('\nLinear SVC Confusion Matrix')
    print(confusion_matrix(Y_test, Y_pred_svc))
    #Plot confusion matrix                            
    fig, ax = plt.subplots(ncols=2, figsize = (20,10), num = title)
    ax1, ax2 = ax
    ax1.set_title('Naive Bayes Confusion Matrix '+title)
    plot_confusion_matrix(model1,X_test_tf,
                      Y_test, values_format='d',
                      cmap=plt.cm.Blues,ax=ax1)
    ax2.set_title('Linear SVC Confusion Matrix '+title)
    plot_confusion_matrix(model2,X_test_tf,
                      Y_test, values_format='d',
                      cmap=plt.cm.Blues,ax=ax2)
    plt.tight_layout()
    plt.show()

#min_df = 10, ngram_range=(1,2)
compareNBwithSVC(10,(1,2))

#min_df = 5, ngram_range=(1,1)
compareNBwithSVC(5,(1,1))

#min_df = 5, ngram_range=(1,3)
compareNBwithSVC(5,(1,3))

#min_df = 1, ngram_range=(1,1)
compareNBwithSVC(1,(1,1))

#min_df = 1, ngram_range=(1,3)
compareNBwithSVC(1,(1,3))