# # Classification using Scikit-Learn on IRIS data
# Author: BORIS KUNDU

#Import packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier

#Read data
data = pd.read_csv('IRIS.csv')
print(f'Head:\n{data.head()}')
print(f'Features:\n{(data.columns)}')

# ## Rename features
data = data.rename(columns={'sepal_length':'X1', 'sepal_width':'X2', 'petal_length':'X3', 'petal_width':'X4','species':'L'})


# ## Change the labels to numbers
#Now let's form a label Encoder model
le = preprocessing.LabelEncoder()
#Now we use feed the label column to the model
le.fit(data['L'])
#Model will go through column and find the unique labels (Number of classes that are there)
#Following line will print the labels found in the column
print(f'Classes:{list(le.classes_)}')

#Following line will convert the labels to an array of numbers
#Lets replace these numbers with the labels in data
data['L'] = le.transform(data['L'])

print(f'Check Head:\n{data.head()}')

# Define feature pairs for plots.
# Create all pairs
feature_pairs = [('X1','X2'),('X1','X3'),('X1','X4'),('X2','X3'),('X2','X4'),('X3','X4')]

# ## Visualize Data
#Loop and plot pairs of features
for (x,y) in feature_pairs:
    plt.figure()
    plt.scatter(data[x][data['L']==0], data[y][data['L']==0],label='Iris-setosa',color='blue')
    plt.scatter(data[x][data['L']==1], data[y][data['L']==1],label='Iris-versicolor',color='red')
    plt.scatter(data[x][data['L']==2], data[y][data['L']==2],label='Iris-virginica',color='green')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'{x} vs {y}')
    plt.legend()
    plt.show()

# ## Split Data into Train and Test
X_training, X_testing, Y_training, Y_testing = train_test_split(data[['X1','X2','X3','X4']], data['L'], test_size=0.3)

#Loop and plot pairs of features for training and testing data
for (x,y) in feature_pairs:
    plt.figure()
    plt.scatter(X_training[x][Y_training==0],X_training[y][Y_training==0],label='Iris-setosa',color='blue')
    plt.scatter(X_training[x][Y_training==1],X_training[y][Y_training==1],label='Iris-versicolor',color='red')
    plt.scatter(X_training[x][Y_training==2],X_training[y][Y_training==2],label='Iris-virginica',color='green')
    plt.title(f'Training Data - {x} vs {y}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.show()
    plt.figure()
    plt.scatter(X_testing[x][Y_testing==0],X_testing[y][Y_testing==0],label='Iris-setosa',color='blue')
    plt.scatter(X_testing[x][Y_testing==1],X_testing[y][Y_testing==1],label='Iris-versicolor',color='red')
    plt.scatter(X_testing[x][Y_testing==2],X_testing[y][Y_testing==2],label='Iris-virginica',color='green')
    plt.title(f'Testing Data - {x} vs {y}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.show()

# ## MDC Classifier
# Classify data using MDC and count the number of points that are missclassified in the training data and test data.
# Create and fit MDC model
mdc = NearestCentroid()
mdc.fit(X_training,Y_training)

#Predict classes for Training and Testing datasets
Y_training_predictions = mdc.predict(X_training)
Y_testing_predictions = mdc.predict(X_testing)

# Evaluating Model for Training Data
# Loop and plot pairs of features for evaluating model on training data
for (x,y) in feature_pairs:
    plt.figure()
    plt.scatter(X_training[x][Y_training==0],X_training[y][Y_training==0], label ='Iris-setosa', color = 'blue' )
    plt.scatter(X_training[x][Y_training==1],X_training[y][Y_training==1], label ='Iris-versicolor', color = 'red' )
    plt.scatter(X_training[x][Y_training==2],X_training[y][Y_training==2], label ='Iris-virginica', color = 'green' )
    plt.scatter(X_training[x][Y_training==0].mean(),X_training[y][Y_training==0].mean(), marker='x', color = 'black' )
    plt.scatter(X_training[x][Y_training==1].mean(),X_training[y][Y_training==1].mean(), marker='x', color = 'black' )
    plt.scatter(X_training[x][Y_training==2].mean(),X_training[y][Y_training==2].mean(), marker='x', color = 'black' )
    plt.scatter(X_training[x][Y_training!=Y_training_predictions],X_training[y][Y_training!=Y_training_predictions], label ='Misclassifications', color = 'yellow' )
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'MDC Classification Result - Training Data - {x} vs {y}')
    plt.legend()
    plt.show()

# Evaluating Model for Testing Data
# Loop and plot pairs of features for evaluating model on testing data
for (x,y) in feature_pairs:
    plt.figure()
    plt.scatter(X_testing[x][Y_testing==0],X_testing[y][Y_testing==0], label ='Iris-setosa', color = 'blue' )
    plt.scatter(X_testing[x][Y_testing==1],X_testing[y][Y_testing==1], label ='Iris-versicolor', color = 'red' )
    plt.scatter(X_testing[x][Y_testing==2],X_testing[y][Y_testing==2], label ='Iris-virginica', color = 'green' )
    plt.scatter(X_testing[x][Y_testing==0].mean(),X_testing[y][Y_testing==0].mean(), marker='x', color = 'black' )
    plt.scatter(X_testing[x][Y_testing==1].mean(),X_testing[y][Y_testing==1].mean(), marker='x', color = 'black' )
    plt.scatter(X_testing[x][Y_testing==2].mean(),X_testing[y][Y_testing==2].mean(), marker='x', color = 'black' )
    plt.scatter(X_testing[x][Y_testing!=Y_testing_predictions],X_testing[y][Y_testing!=Y_testing_predictions], label ='Misclassifications', color = 'yellow' )
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'MDC Classification Result - Testing Data - {x} vs {y}')
    plt.legend()
    plt.show()

# MDC Errors
# Training Errors
trainErrors = 0
trainDiff = Y_training - Y_training_predictions
for diff in trainDiff:
    if diff != 0:
        trainErrors = trainErrors + 1
print(f'MDC Training Data Misclassifications: {trainErrors}')
#Testing Errors
testErrors = 0
testingDiff = Y_testing - Y_testing_predictions
for diff in testingDiff:
    if diff != 0:
        testErrors = testErrors + 1
print(f'MDC Testing Data Misclassifications: {testErrors}')

# ## K-Nearst Neighbors Classifier
# Classify data using KNN (K=5 nearest neighbors) and count the number of points that are missclassified in the training data and test data.
# Create and fit KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_training,Y_training)

#Predict classes for Training and Testing datasets
Y_training_prediction = knn.predict(X_training)
Y_testing_prediction = knn.predict(X_testing)

#Loop and plot pairs of features for evaluating model on training data
for (x,y) in feature_pairs:
    plt.figure()
    plt.scatter(X_training[x][Y_training==0],X_training[y][Y_training==0], label ='Iris-setosa', color = 'blue' )
    plt.scatter(X_training[x][Y_training==1],X_training[y][Y_training==1], label ='Iris-versicolor', color = 'red' )
    plt.scatter(X_training[x][Y_training==2],X_training[y][Y_training==2], label ='Iris-virginica', color = 'green' )
    plt.scatter(X_training[x][Y_training==0].mean(),X_training[y][Y_training==0].mean(), marker='x', color = 'black' )
    plt.scatter(X_training[x][Y_training==1].mean(),X_training[y][Y_training==1].mean(), marker='x', color = 'black' )
    plt.scatter(X_training[x][Y_training==2].mean(),X_training[y][Y_training==2].mean(), marker='x', color = 'black' )
    plt.scatter(X_training[x][Y_training!=Y_training_prediction],X_training[y][Y_training!=Y_training_prediction], label ='Misclassifications', color = 'yellow' )
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'KNN Classification Result - Training Data - {x} vs {y}')
    plt.legend()
    plt.show()

# Loop and plot pairs of features for evaluating model on testing data
for (x,y) in feature_pairs:
    plt.figure()
    plt.scatter(X_testing[x][Y_testing==0],X_testing[y][Y_testing==0], label ='Iris-setosa', color = 'blue' )
    plt.scatter(X_testing[x][Y_testing==1],X_testing[y][Y_testing==1], label ='Iris-versicolor', color = 'red' )
    plt.scatter(X_testing[x][Y_testing==2],X_testing[y][Y_testing==2], label ='Iris-virginica', color = 'green' )
    plt.scatter(X_testing[x][Y_testing==0].mean(),X_testing[y][Y_testing==0].mean(), marker='x', color = 'black' )
    plt.scatter(X_testing[x][Y_testing==1].mean(),X_testing[y][Y_testing==1].mean(), marker='x', color = 'black' )
    plt.scatter(X_testing[x][Y_testing==2].mean(),X_testing[y][Y_testing==2].mean(), marker='x', color = 'black' )
    plt.scatter(X_testing[x][Y_testing!=Y_testing_prediction],X_testing[y][Y_testing!=Y_testing_prediction], label ='Misclassifications', color = 'yellow' )
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'KNN Classification Result - Testing Data - {x} & {y}')
    plt.legend()
    plt.show()

# KNN Errors
# Training Errors
trainError = 0
trainDiff = Y_training - Y_training_prediction
for diff in trainDiff:
    if diff != 0:
        trainError = trainError + 1
print(f'KNN Training Data Misclassifications: {trainError}')
#Testing Errors
testError = 0
testingDiff = Y_testing - Y_testing_prediction
for diff in testingDiff:
    if diff != 0:
        testError = testError + 1
print(f'KNN Testing Data Misclassifications: {testError}')