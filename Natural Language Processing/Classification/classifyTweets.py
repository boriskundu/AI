# Author: Boris Kundu
# Predict tweet year from content using Linear Support Vector Classifier
# Compare results with a Dummy Classifer

#Import packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix  
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt

#Read data
df = pd.read_csv('tweets.csv')
#Create target class 'year' from date
df['year'] = (pd.to_datetime(df['date'])).dt.year
#Plot bar chart for 'year' class
df['year'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Class Size Bar Chart')
plt.show()
#Split data
X_train, X_test, Y_train, Y_test = train_test_split(df['text'],
                                                    df['year'],
                                                    test_size = 0.3,
                                                    stratify = df['year'])
#Check size
print('Size of Training Data ', X_train.shape[0])
print('Size of Test Data ', X_test.shape[0])
#Get tfidf
tfidf = TfidfVectorizer(min_df = 10, ngram_range=(1,2), stop_words="english")
X_train_tf = tfidf.fit_transform(X_train)
#Train linear SVC
model1 = LinearSVC(random_state=0, tol=1e-5)
model1.fit(X_train_tf, Y_train)

X_test_tf = tfidf.transform(X_test)
#Predict tweet year
Y_pred = model1.predict(X_test_tf)
#Print accuracy
print ('LinearSVC Accuracy Score - ', accuracy_score(Y_test, Y_pred))

#Train and predict using a dummy classifier
clf = DummyClassifier(strategy='most_frequent')
clf.fit(X_train, Y_train)
Y_pred_baseline = clf.predict(X_test)
print ('DummyClassifier Accuracy Score - ', accuracy_score(Y_test, Y_pred_baseline))

#Show confusiin marix
print(f'Linear SVC Confusion Matrix:\n{confusion_matrix(Y_test, Y_pred)}')

#Plot confusion matrix for 'year' class predictions
fig, ax = plt.subplots(figsize = (20,10))
ax.set_title('Confusion Matrix')
plot_confusion_matrix(model1,X_test_tf, Y_test, values_format='d', cmap=plt.cm.Blues,ax=ax)
plt.tight_layout()
plt.show()