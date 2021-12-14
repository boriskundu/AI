#!/usr/bin/env python
# coding: utf-8

# ***Linear Regression using Normal Equation for insurance data.***
# 
# **Author: BORIS KUNDU**

#Import libraries 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

#Read insurance.csv
insuranceData = pd.read_csv('insurance.csv')

#Check data head
print(f'Sample Data:\n{insuranceData.head()}')

#Check data info
print(f'Data Info:\n{insuranceData.info()}')

#Describe data
print(f'Data Stats:\n{insuranceData.describe()}')

#Perform Exploratory Data Analysis (EDA)
#Check feature pair plot
sns.pairplot(insuranceData)

#Check feature corelation heatmap
plt.figure()
sns.heatmap(insuranceData.corr(),cmap='coolwarm')

#Eliminate categorical features for our data
insuranceData.drop(['sex','smoker','region'],axis=1)

#Define training features in X
X = insuranceData[['age','bmi','children']]

#Adding bias term/extra feature as 1s to X
X['bias'] = 1

#Define target ouput in y
y = insuranceData[['charges']]

#Function to make predictions
def getPredictions(X_train,y_train,X_test):
    #Calculate theta using Train set features and Train target output (Charges).
    theta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
    print(f'\nTheta from training data:{theta}')
    #Predict Train & Test target ouput (Charges) using features and Theta
    predictTestCharges = np.dot(X_test, theta)
    predictTrainCharges = np.dot(X_train, theta)
    return (predictTestCharges,predictTrainCharges)

#Function to calculate Mean Square Error
def getMSE(predictTestCharges,y_test,predictTrainCharges,y_train):
    trainMSE = metrics.mean_squared_error(y_train, predictTrainCharges)
    testMSE = metrics.mean_squared_error(y_test, predictTestCharges)
    print('\nGeneralization Power - Test Set MSE:',testMSE)
    print('Modeling Power - Train Set MSE:',trainMSE)
    return (testMSE,trainMSE)

#Function to display plots
def display(predictTestCharges, predictTrainCharges, testMSE, trainMSE, X_train,y_train, X_test,y_test,t):
    fig = plt.figure(figsize = [10,5])
    
    # BMI vs Charges
    plt.subplot(1,3,1)
    plt.scatter(X_test['bmi'], y_test,label='Expected')
    plt.scatter(X_test['bmi'], predictTestCharges,label='Actual')
    plt.xlabel('BMI')
    plt.ylabel('Premium')
    plt.title(f'BMI vs Premium - Training Size:{t*100}%')
    plt.legend()
    
    # Age vs Charges
    plt.subplot(1,3,2)
    plt.scatter(X_test['age'], y_test, label ='Expected')
    plt.scatter(X_test['age'], predictTestCharges, label ='Actual')
    plt.xlabel('Age')
    plt.ylabel('Premium')
    plt.title(f'Age vs Premium - Training Size:{t*100}%')
    plt.legend()

    # Children vs Charges
    plt.subplot(1,3,3)
    plt.scatter(X_test['children'], y_test, label ='Expected')
    plt.scatter(X_test['children'], predictTestCharges,label ='Actual')
    plt.xlabel('Children')
    plt.ylabel('Premium')
    plt.title(f'Children vs Premium - Training Size:{t*100}%')
    plt.legend()

    plt.tight_layout()
    plt.show()
    print(f'\n*** End running linear regression with train size:{t*100}% ***')

#Define function to run linear regression
#Takes training data % as input
def runLinearRegression(t):
    print(f'\n*** Start running linear regression with train size:{t*100}% ***')
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=t, random_state=101)
    #Get theta (weights) using Normal equation
    predictTestCharges,predictTrainCharges = getPredictions(X_train,y_train,X_test)
    testMSE,trainMSE = getMSE(predictTestCharges,y_test,predictTrainCharges,y_train)
    display(predictTestCharges, predictTrainCharges, testMSE, trainMSE, X_train,y_train, X_test, y_test,t)
    a.append(testMSE)
    b.append(trainMSE)

a = []
b = []
#Run regression with test set size as 50%,60%,70% and 80% respectively.
for t in [0.5,0.6,0.7,0.8]:
    runLinearRegression(t)