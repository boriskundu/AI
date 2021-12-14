#!/usr/bin/env python
# coding: utf-8

# **Linear Regression - Gradient Descent**
# 
# Author - Boris Kundu

#Import packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import copy
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

#Read CSV data
df = pd.read_csv('insurance.csv')

#Check corelation heatmap for numeric features
#1 indicates max corelation
#0 indicates no corelation
dfNum = df[['age','bmi','children','charges']]

#Eliminate Region from our data
df.drop('region',axis=1,inplace=True)

#Define training features in X
X = df[['age','sex','bmi','children','smoker']]
#Define output target feature in y
y = df[['charges']]

#Transform feature sex by changing female to 0 and male to 1
X['sex'].replace(['female','male'],[0,1],inplace=True)
#Transform feature smoker by changing no to 0 and yes to 1
X['smoker'].replace(['no','yes'],[0,1],inplace=True)

#Standardize numerical features of X and y 
XN = (X-np.mean(X))/np.std(X)
yN = (y-np.mean(y))/np.std(y)

#Add 1s as additional feature called bias in normalized X
XN['bias'] = np.ones(len(yN))

#Split data randomly into Train and Test
X_train, X_test, y_train, y_test = train_test_split(XN, yN, train_size=0.7, random_state=101)

#Split the training data into chnks of size of the mini-batch 
mini_batch_size=4
X_train_mini=np.array_split(X_train,mini_batch_size)
Y_train_mini=np.array_split(y_train,mini_batch_size)

#Function returns best posible alpha for L2 Regularization using KFold
def parameterTuning(X,y):
    #Range of alpha values to check
    alphas = np.linspace(start = 0, stop = 10,num = 50)
    # Start with 50%
    bestAccuracy = 50 
    bestAlpha = 0
    accuracy = 0
    
    #Perform K-Fold cross validation to get the alpha that gives the highest accuracy.
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        kfold = KFold(n_splits=10)
        results = cross_val_score(model,X,y,cv=kfold)
        accuracy = results.mean()*100
        #print(f'Accuracy:{accuracy}')
        if (accuracy >= bestAccuracy):
            bestAccuracy = accuracy
            bestAlpha = alpha
    print(f'Best alpha for L2 Ridge Regularization:{bestAlpha} giving accuracy:{bestAccuracy}')
    return bestAlpha

def display(X_test,y_test,w,fig_name):
    print('Displaying scatter and line plots for all features ...')
    #Predict charges by using best weights on test dataset
    y_predicted_test = np.dot(X_test,w)
    for f in X_test.columns:
        if f!='bias' and f!='sex' and f!='children':
            plt.figure(figsize=(8,4),num=fig_name)
            plt.title('Regression Line')
            plt.xlabel(f)
            plt.ylabel('Premium')
            feature = X_test[f].to_numpy()
            m, b = np.polyfit(feature, y_predicted_test, 1)
            m_ex,b_ex = np.polyfit(feature, y_test, 1)
            plt.plot(feature,m_ex*feature+b_ex,label = 'Expected')
            plt.plot(feature,m*feature+b,color='orange',label = 'Actual')
            plt.legend()
            plt.show()

#Function to calculate and plot Gradient Descent without Regularization
def gradientDescentBatchNR(X_train,y_train,X_test,y_test,eta=0.05,looped=100,acceptableError=0.1):
    print('RUNNING BATCH GRADIENT DESCENT WITHOUT REGULARIZATION ...')
    #Least Cost
    leastCost = 1000
    #MSE aka cost
    MSE_train = []
    #Size of training set aka Rows
    m=len(y_train)
    #Size of features (including bias) aka Columns
    n=len(X_train.columns)
    #Define Weight Matrix and randomly initliaze it
    w=np.random.randint(0,6,n).reshape(n,1)
    #Best Weights
    bestWeights = w
    #Calculate predicted y aka hypothesis
    for loop in range(looped):
        #Calculate hypothesis/predicted values
        y_hat = np.dot(X_train,w)
        #Calculate loss in Training data
        loss_train = y_hat - y_train
        #Calculcate cost aka MSE
        cost_train = float((1/m)*(np.sum(loss_train ** 2)))
        #print(f'Loop:{loop+1} => Cost:{cost_train}')
        # MSE list
        MSE_train.append(cost_train) 
        #Transpose X_train
        X_train_T = X_train.T
        #Calculate Gradient vector
        gradMSEw = (2/m) * (np.dot(X_train_T,loss_train))
        #Calculate least cost and update bestWeights
        if(cost_train <= leastCost):
            leastCost = cost_train
            bestWeights = w
        #Update weights using Training Rate using eta
        w = w - eta*gradMSEw
        #Check if we have reached the expected
        if (cost_train <= acceptableError):
            break
    #Show least Cost and Best Weight
    print(f'Least Cost = {leastCost} and Best Weights = {bestWeights}')

    #Plot for Actual vs Predicted Test Data for all features
    display(X_test,y_test,bestWeights,'BATCH GRADIENT DESCENT')

#Function to calculate and plot Gradient Descent with L1 Regularization
def gradientDescentBatchL1(X_train,y_train,X_test,y_test,eta=0.05,looped=100,acceptableError=0.1):
    print('RUNNING BATCH GRADIENT DESCENT WITH L1 REGULARIZATION ...')
    #Set L1 alpha
    alpha = len(X_train.columns) - 1
    #Least Cost
    leastCost = 1000
    #MSE aka cost
    MSE_train = []
    #Size of training set aka Rows
    m=len(y_train)
    #Size of features (including bias) aka Columns
    n=len(X_train.columns)
    #Define Weight Matrix and randomly initliaze it
    w=np.random.randint(0,6,n).reshape(n,1)
    #Best Weights
    bestWeights = w
    #Clculate predicted y aka hypothesis
    for loop in range(looped):
        #Calculate hypothesis/predicted values
        y_hat = np.dot(X_train,w)
        #Calculate loss in Training data
        loss_train = y_hat - y_train
        #Calculcate cost aka MSE
        cost_train = float((1/m)*(np.sum(loss_train ** 2)))
        #print(f'Loop:{loop+1} => Cost:{cost_train}')
        # MSE list
        MSE_train.append(cost_train) 
        #Transpose X_train
        X_train_T = X_train.T
        #Calculate Gradient vector
        gradMSEw = (2/m) * (np.dot(X_train_T,loss_train)) + 2*alpha*np.sign(w)
        #Calculate least cost and update bestWeights
        if(cost_train <= leastCost):
            leastCost = cost_train
            bestWeights = w
        #Update weights using L1 Regularization (Lasso)
        w = w - eta*gradMSEw
        #Check if we have reached the expected
        if (cost_train <= acceptableError):
            break
    #Show least Cost and Best Weight
    print(f'Least Cost = {leastCost} and Best Weights = {bestWeights}')
    
    #Drop features with weight almost zero from both Training and Test data
    pos = 0
    dropColumns = []
    min_wt = np.min(bestWeights)
    for wt,pos in zip(bestWeights,range(len(bestWeights))):
        if min_wt == wt[0]:
            dropColumns.append(pos)
            print(f'We should drop this feature at position {pos+1} having weight {wt[0]}')
    print(f'Dropping feature at index positions:{dropColumns}')
    print(f'Dropping feature : {X_train.columns[dropColumns]}')
    
    #Dropping Features from Training Data
    X_train.drop(X_train.columns[dropColumns],axis = 1, inplace=True)
    X_test.drop(X_test.columns[dropColumns],axis = 1, inplace=True)
    print(f'Remaining features:{X_train.columns}')
    #Now calling Batch GD after performing L1 regularization
    print(f'Calling Batch GD with reduced features..')
    gradientDescentBatchNR(X_train,y_train,X_test,y_test,eta,looped,acceptableError)

#Function to calculate and plot Gradient Descent with L2 Regularization
def gradientDescentBatchL2(X_train,y_train,X_test,y_test,eta=0.05,looped=100,acceptableError=0.1):
    print('RUNNING BATCH GRADIENT DESCENT WITH L2 REGULARIZATION ...')
    #Get Best alpha for L2 regularization
    alpha = parameterTuning(XN.copy(),yN.copy())
    #Least Cost
    leastCost = 1000
    #MSE aka cost
    MSE_train = []
    #Size of training set aka Rows
    m=len(y_train)
    #Size of features (including bias) aka Columns
    n=len(X_train.columns)
    #Define Weight Matrix and randomly initliaze it
    w=np.random.randint(0,6,n).reshape(n,1)
    #Best Weights
    bestWeights = w
    #Clculate predicted y aka hypothesis
    for loop in range(looped):
        #Calculate hypothesis/predicted values
        y_hat = np.dot(X_train,w)
        #Calculate loss in Training data
        loss_train = y_hat - y_train
        #Calculcate cost aka MSE
        cost_train = float((1/m)*(np.sum(loss_train ** 2)))
        # print(f'Loop:{loop+1} => Cost:{cost_train}')
        # MSE list
        MSE_train.append(cost_train) 
        #Transpose X_train
        X_train_T = X_train.T
        #Calculate Gradient vector
        gradMSEw = (2/m) * (np.dot(X_train_T,loss_train)) + alpha*w
        #Calculate least cost and update bestWeights
        if(cost_train <= leastCost):
            leastCost = cost_train
            bestWeights = w
        #Update weights using L2 Regularization (Ridge)
        w = w - eta*gradMSEw
        #Check if we have reached the expected
        if (cost_train <= acceptableError):
            break
    #Show least Cost and Best Weight
    print(f'Least Cost = {leastCost} and Best Weights = {bestWeights}')

    #Drop features with weight almost zero from both Training and Test data
    pos = 0
    dropColumns = []
    min_wt = np.min(bestWeights)
    for wt,pos in zip(bestWeights,range(len(bestWeights))):
        if min_wt == wt[0]:
            dropColumns.append(pos)
            print(f'We should drop this feature at position {pos+1} having weight {wt[0]}')
    print(f'Dropping feature at index positions:{dropColumns}')
    print(f'Dropping feature : {X_train.columns[dropColumns]}')
    
    #Dropping Features from Training & Test Data
    X_train.drop(X_train.columns[dropColumns],axis = 1, inplace=True)
    X_test.drop(X_test.columns[dropColumns],axis = 1, inplace=True)
    print(f'Remaining features:{X_train.columns}')
    #Now calling Batch GD after performing L1 regularization
    print(f'Calling Batch GD with reduced features..')
    gradientDescentBatchNR(X_train,y_train,X_test,y_test,eta,looped,acceptableError)


#Call Batch GD without regularization
gradientDescentBatchNR(X_train.copy(),y_train.copy(),X_test.copy(),y_test.copy(),0.01,1000,0.1)

#Call Batch GD with L1 regularization
gradientDescentBatchL1(X_train.copy(),y_train.copy(),X_test.copy(),y_test.copy(),0.01,1000,0.1)

#Call Batch GD with L2 regularization
gradientDescentBatchL2(X_train.copy(),y_train.copy(),X_test.copy(),y_test.copy(),0.01,1000,0.1)

#Function to calculate least cost and best weights for Mini Batch Gradient Descent
def gradientDescentMiniBatchNR(X_train,y_train,eta=0.05,looped=100,acceptableError=0.1):
    #Least Cost
    leastCost = 1000
    #MSE aka cost
    MSE_train = []
    #Size of training set aka Rows
    m=len(y_train)
    #Size of features (including bias) aka Columns
    n=len(X_train.columns)
    #Define Weight Matrix and randomly initliaze it
    w=np.random.randint(0,6,n).reshape(n,1)
    #Best Weights
    bestWeights = w
    #Calculate predicted y aka hypothesis
    for loop in range(looped):
        #Calculate hypothesis/predicted values
        y_hat = np.dot(X_train,w)
        #Calculate loss in Training data
        loss_train = y_hat - y_train
        #Calculcate cost aka MSE
        cost_train = float((1/m)*(np.sum(loss_train ** 2)))
        #print(f'Loop:{loop+1} => Cost:{cost_train}')
        # MSE list
        MSE_train.append(cost_train) 
        #Transpose X_train
        X_train_T = X_train.T
        #Calculate Gradient vector
        gradMSEw = (2/m) * (np.dot(X_train_T,loss_train))
        #Calculate least cost and update bestWeights
        if(cost_train <= leastCost):
            leastCost = cost_train
            bestWeights = w
        #Update weights using Training Rate using eta
        w = w - eta*gradMSEw
        #Check if we have reached the expected
        if (cost_train <= acceptableError):
            break
    #Show least Cost and Best Weight
    print(f'Least Cost = {leastCost} and Best Weights = {bestWeights}')
    return (leastCost,bestWeights)

#Call Mini Batch GD without regularization
def Mini_Batch_GD(X_train_mini,Y_train_mini,X_test,y_test):
    print('RUNNING MINI BATCH GRADIENT DESCENT WITHOUT REGULARIZATION ...')
    leastCost_sum=0
    w_sum=0
    for i in range(len(X_train_mini)):
        print(f'Calling Mini Batch for set {i+1}')
        leastCost_mini,w_mini=gradientDescentMiniBatchNR(X_train_mini[i],Y_train_mini[i],0.01,1000,0.1)
        leastCost_sum+=leastCost_mini
        w_sum+=w_mini
    #Show the average least Cost and Best Weight
    w = w_sum/mini_batch_size
    avgCost = leastCost_sum/mini_batch_size
    print(f'Average Least Cost = {avgCost} and Average Best Weights = {w}')
    #Display predictions using avergae weights
    print(f'Predicting target using best average weights')
    display(X_test,y_test,w,'MINI BATCH GRADIENT DESCENT')

#Call Mini Batch without regularization
Mini_Batch_GD(copy.deepcopy(X_train_mini),copy.deepcopy(Y_train_mini),X_test.copy(),y_test.copy())

#Function to calculate and plot Mini Batch Gradient Descent with L1 Regularization
def gradientDescentMiniBatchL1(X_train_mini_L1,Y_train_mini_L1,eta=0.05,looped=100,acceptableError=0.1,alpha = 5):
    #Least Cost
    leastCost = 1000
    #MSE aka cost
    MSE_train = []
    #Size of training set aka Rows
    m=len(Y_train_mini_L1)
    #Size of features (including bias) aka Columns
    n=len(X_train_mini_L1.columns)
    #Define Weight Matrix and randomly initliaze it
    w=np.random.randint(0,6,n).reshape(n,1)
    #Best Weights
    bestWeights = w
    #Clculate predicted y aka hypothesis
    for loop in range(looped):
        #Calculate hypothesis/predicted values
        y_hat = np.dot(X_train_mini_L1,w)
        #Calculate loss in Training data
        loss_train = y_hat - Y_train_mini_L1
        #Calculcate cost aka MSE
        cost_train = float((1/m)*(np.sum(loss_train ** 2)))
        #print(f'Loop:{loop+1} => Cost:{cost_train}')
        # MSE list
        MSE_train.append(cost_train) 
        #Transpose X_train
        X_train_mini_L1_T = X_train_mini_L1.T
        #Calculate Gradient vector
        gradMSEw = (2/m) * (np.dot(X_train_mini_L1_T,loss_train)) + 2*alpha*np.sign(w)
        #Calculate least cost and update bestWeights
        if(cost_train <= leastCost):
            leastCost = cost_train
            bestWeights = w
        #Update weights using L1 Regularization (Lasso)
        w = w - eta*gradMSEw
        #Check if we have reached the expected
        if (cost_train <= acceptableError):
            break
    #Show least Cost and Best Weight
    print(f'Least Cost = {leastCost} and Best Weights = {bestWeights}')
    return (leastCost,bestWeights)

#Function for Mnini Batch L1
def Mini_Batch_L1(X_train_mini_L1,Y_train_mini_L1,X_test_L1,y_test_L1):
    print('RUNNING MINI BATCH GRADIENT DESCENT WITH L1 REGULARIZATION ...')
    #Call Mini Batch GD with L1 regularization
    leastCost_sum=0
    w_sum=0
    #Set L1 alpha to number of features excluding bias
    alpha = len(X_test.columns) - 1
    for i in range(len(X_train_mini_L1)):
        print(f'Calling Mini Batch GD with L1 regularization for set {i+1}')
        leastCost_mini,w_mini=gradientDescentMiniBatchL1(X_train_mini_L1[i],Y_train_mini[i],0.01,1000,0.1,alpha)
        leastCost_sum+=leastCost_mini
        w_sum+=w_mini
    
    w = w_sum/mini_batch_size
    avgCost = leastCost_sum/mini_batch_size
    #Show the average least Cost and Best Weight
    print(f'Before feature reduction Average Least Cost = {avgCost} and Average Best Weights = {w}')
    
    #Drop feature with least weight from both Training and Test data
    pos = 0
    dropColumns = []
    min_wt = np.min(w)
    for wt,pos in zip(w,range(len(w))):
        if min_wt == wt[0]:
            dropColumns.append(pos)
            print(f'We should drop this feature at position {pos+1} having weight {wt[0]}')
    print(f'Dropping feature at index positions:{dropColumns}')
    print(f'Dropping feature : {X_train_mini_L1[0].columns[dropColumns]}')
    
    #Dropping Features from Training & Test Data
    for i in range(len(X_train_mini_L1)):
        X_train_mini_L1[i].drop(X_train_mini_L1[i].columns[dropColumns],axis = 1, inplace=True)
    print(f'Remaining features:{X_train_mini_L1[0].columns}')
    X_test_L1.drop(X_test_L1.columns[dropColumns],axis = 1, inplace=True)

    #Now calling Mini Batch GD after performing L1 regularization
    print(f'Calling Mini Batch Gradient Descent after L1 regularization ...')
    Mini_Batch_GD(X_train_mini_L1,Y_train_mini_L1,X_test_L1,y_test_L1)

#Call Mini Batch L1
Mini_Batch_L1(copy.deepcopy(X_train_mini),copy.deepcopy(Y_train_mini),X_test.copy(),y_test.copy())

#Function to calculate and plot Mini Batch Gradient Descent with L2 Regularization
def gradientDescentMiniBatchL2(X_train_mini_L2,Y_train_mini_L2,eta=0.05,looped=100,acceptableError=0.1,alpha=5):
    #Least Cost
    leastCost = 1000
    #MSE aka cost
    MSE_train = []
    #Size of training set aka Rows
    m=len(Y_train_mini_L2)
    #Size of features (including bias) aka Columns
    n=len(X_train_mini_L2.columns)
    #Define Weight Matrix and randomly initliaze it
    w=np.random.randint(0,6,n).reshape(n,1)
    #Best Weights
    bestWeights = w
    #Clculate predicted y aka hypothesis
    for loop in range(looped):
        #Calculate hypothesis/predicted values
        y_hat = np.dot(X_train_mini_L2,w)
        #Calculate loss in Training data
        loss_train = y_hat - Y_train_mini_L2
        #Calculcate cost aka MSE
        cost_train = float((1/m)*(np.sum(loss_train ** 2)))
        # print(f'Loop:{loop+1} => Cost:{cost_train}')
        # MSE list
        MSE_train.append(cost_train) 
        #Transpose X_train
        X_train_mini_L2_T = X_train_mini_L2.T
        #Calculate Gradient vector
        gradMSEw = (2/m) * (np.dot(X_train_mini_L2_T,loss_train)) + alpha*w
        #Calculate least cost and update bestWeights
        if(cost_train <= leastCost):
            leastCost = cost_train
            bestWeights = w
        #Update weights using L2 Regularization (Ridge)
        w = w - eta*gradMSEw
        #Check if we have reached the expected
        if (cost_train <= acceptableError):
            break
    #Show least Cost and Best Weight
    print(f'Least Cost = {leastCost} and Best Weights = {bestWeights}')
    return (leastCost,bestWeights)

#Function for Mini Batch L2
def Mini_Batch_L2(X_train_mini_L2,Y_train_mini_L2,X_test_L2,y_test_L2):
    print('RUNNING MINI BATCH GRADIENT DESCENT WITH L2 REGULARIZATION ...')
    #Get Best alpha for Mini Batch L2 regularization
    alpha= [None]*len(X_train_mini_L2) 
    leastCost_sum_L2=0
    w_sum_L2=0
    #Get L2 alpha from KFold
    for i in range(len(X_train_mini_L2)):
        alpha[i] = parameterTuning(X_train_mini_L2[i],Y_train_mini_L2[i])
        #Call Mini Batch GD with L2 regularization
        print(f'Calling Mini Batch GD with L2 regularization for set {i+1}')
        leastCost_l2_mini,w_l2_mini=gradientDescentMiniBatchL2(X_train_mini_L2[i],Y_train_mini_L2[i],0.01,1000,0.1,alpha[i])
        leastCost_sum_L2+=leastCost_l2_mini
        w_sum_L2+=w_l2_mini
    
    #Show the average least Cost and Best Weight
    w = w_sum_L2/mini_batch_size
    avgCost = leastCost_sum_L2/mini_batch_size
    #Show the average least Cost and Best Weight
    print(f'Before feature reduction Average Least Cost = {avgCost} and Average Best Weights = {w}')
    
    #Drop feature with least weight from both Training and Test data
    pos = 0
    dropColumns = []
    min_wt = np.min(w)
    for wt,pos in zip(w,range(len(w))):
        if min_wt == wt[0]:
            dropColumns.append(pos)
            print(f'We should drop this feature at position {pos+1} having weight {wt[0]}')
    print(f'Dropping feature at index positions:{dropColumns}')
    print(f'Dropping feature : {X_train_mini_L2[0].columns[dropColumns]}')
    
    #Dropping Features from Training Data
    for i in range(len(X_train_mini)):
        X_train_mini_L2[i].drop(X_train_mini_L2[i].columns[dropColumns],axis = 1, inplace=True)
    print(f'Remaining features:{X_train_mini_L2[0].columns}')
    X_test_L2.drop(X_test_L2.columns[dropColumns],axis = 1, inplace=True)

    #Now calling Batch GD after performing L2 regularization
    print(f'Calling Mini Batch Gradient Descent after L2 regularization ...')
    Mini_Batch_GD(X_train_mini_L2,Y_train_mini_L2,X_test_L2,y_test_L2)

#Call Mini Batch L2
Mini_Batch_L2(copy.deepcopy(X_train_mini),copy.deepcopy(Y_train_mini),X_test.copy(),y_test.copy())

#Function to calculate and plot Stochastic Gradient Descent without Regularization
def gradientDescentStochasticNR(X_train,y_train, X_test, y_test,eta=0.05,acceptableError=0.1,epoch=100):
    print('RUNNING STOCHASTIC GRADIENT DESCENT WITHOUT REGULARIZATION ...')
    #Least Cost
    leastCost = 1000
    #Size of training set aka Rows
    m=len(y_train)
    #Size of features (including bias) aka Columns
    n=len(X_train.columns)
    #Define Weight Matrix and randomly initialize it
    w=np.random.randint(0,10,n).reshape(n,1)
    #Best Weights
    bestWeights = w
    #Calculate predicted y aka hypothesis
    for e in range(epoch):
        mean_cost = 0
        #MSE aka cost
        MSE_train = []
        for loop in range(m):
            # rand_instance = np.random.randint(0, m)
            #Taking Instance of Training set
            X_rand = X_train.iloc[loop].to_frame()
            Y_rand = y_train.iloc[loop].to_frame()
            #Calculate hypothesis/predicted values
            y_hat = np.dot(X_rand.T,w)
            #Calculate loss in Training data
            loss_train = y_hat - Y_rand
            #Calculcate cost aka MSE
            cost_train = float(np.sum(loss_train ** 2))
            #print(f'Loop:{loop+1} => Cost:{cost_train}')
            # MSE list
            MSE_train.append(cost_train) 
            #Transpose X_train
            #Calculate Gradient vector
            gradMSEw = (2) * (np.dot(X_rand,loss_train))
            #Update weights using gradient and eta
            w = w - eta*gradMSEw
        mean_cost = sum(MSE_train)/len(MSE_train)
        #Calculate least cost and update bestWeights
        if(mean_cost <= leastCost):
            leastCost = cost_train
            bestWeights = w
            eta = eta/1.02 #Add learning decay
        if (mean_cost <= acceptableError):
            break
    #Show least Cost and Best Weight
    print(f'Least Cost = {leastCost} and Best Weights = {bestWeights}')

    #Plot for Actual vs Predicted Test Data for all features
    display(X_test,y_test,bestWeights,'STOCHASTIC GRADIENT DESCENT')

#Call Stochastic GD without regularization
gradientDescentStochasticNR(X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy(),0.01,0.1,10)

#Function to calculate and plot Gradient Descent without Regularization
def gradientDescentStochasticL1(X_train, y_train, X_test, y_test,eta=0.05,acceptableError=0.1,epoch=100):
    print('RUNNING STOCHASTIC GRADIENT DESCENT WITH L1 REGULARIZATION ...')
    #Set alpha to non bias feature count
    alpha = len(X_train.columns) - 1
    #Least Cost
    leastCost = 1000
    #Size of training set aka Rows
    m=len(y_train)
    #Size of features (including bias) aka Columns
    n=len(X_train.columns)
    #Define Weight Matrix and randomly initliaze it
    w=np.random.randint(0,10,n).reshape(n,1)
    #Best Weights
    bestWeights = w
    #Clculate predicted y aka hypothesis
    for e in range(epoch):
        mean_cost = 0
        #MSE aka cost
        MSE_train = []
        for loop in range(m):
            #Taking Instance of Training set
            X_rand = X_train.iloc[loop].to_frame()
            Y_rand = y_train.iloc[loop].to_frame()
            #Calculate hypothesis/predicted values
            y_hat = np.dot(X_rand.T,w)
            #Calculate loss in Training data
            loss_train = y_hat - Y_rand
            #Calculcate cost aka MSE
            cost_train = float(np.sum(loss_train ** 2))
            #print(f'Loop:{loop+1} => Cost:{cost_train}')
            # MSE list
            MSE_train.append(cost_train) 
            #Calculate least cost and update bestWeights
            gradMSEw = 2*(np.dot(X_rand,loss_train)) + 2*alpha*np.sign(w)
            #Update weights using L1 Regularization (Lasso)
            w = w - eta*gradMSEw
        mean_cost = sum(MSE_train)/len(MSE_train)
        #Calculate least cost and update bestWeights
        if(mean_cost <= leastCost):
            leastCost = cost_train
            bestWeights = w
            eta = eta/1.02 #Add learning decay
        if (mean_cost <= acceptableError):
            break
    #Show least Cost and Best Weight
    print(f'Least Cost = {leastCost} and Best Weights = {bestWeights}')
    #Drop features with weight almost zero from both Training data
    dropColumns = []
    min_wt = np.min(bestWeights)
    for wt,pos in zip(bestWeights,range(len(bestWeights))):
        if min_wt == wt[0]:
            dropColumns.append(pos)
            print(f'We should drop this feature at position {pos+1} having weight {wt[0]}')
    print(f'Dropping feature at index positions:{dropColumns}')
    print(f'Dropping feature : {X_train.columns[dropColumns]}')
    #Dropping Features from Training Data
    X_train.drop(X_train.columns[dropColumns],axis = 1, inplace=True)
    X_test.drop(X_test.columns[dropColumns],axis = 1, inplace=True)
    print(f'Remaining features:{X_train.columns}')
    print(f'Calling Stochastic Gradient Descent after L1 regularization ...')
    #Now calling Stochastic GD after performing L1 regularization
    gradientDescentStochasticNR(X_train,y_train,X_test,y_test,eta,acceptableError,epoch)

#Call Stochastic L1 GD
gradientDescentStochasticL1(X_train.copy(),y_train.copy(), X_test.copy(), y_test.copy(),0.01,0.1,10)

#Function to calculate and plot Gradient Descent with L2 Regularization
def gradientDescentStochasticL2(X_train,y_train,X_test,y_test,eta=0.05,acceptableError=0.1,epoch=100):
    print('RUNNING STOCHASTIC GRADIENT DESCENT WITH L2 REGULARIZATION ...')
    #Get Best alpha for L2 regularization
    alpha = parameterTuning(XN.copy(),yN.copy())
    #Least Cost
    leastCost = 1000
    #Size of training set aka Rows
    m=len(y_train)
    #Size of features (including bias) aka Columns
    n=len(X_train.columns)
    #Define Weight Matrix and randomly initliaze it
    w=np.random.randint(0,6,n).reshape(n,1)
    #Best Weights
    bestWeights = w
    #Clculate predicted y aka hypothesis
    for e in range(epoch):
        mean_cost = 0
        #MSE aka cost
        MSE_train = []
        for loop in range(m):
            #Taking Instance of Training set
            X_rand = X_train.iloc[loop].to_frame()
            Y_rand = y_train.iloc[loop].to_frame()
            #Calculate hypothesis/predicted values
            y_hat = np.dot(X_rand.T,w)
            #Calculate loss in Training data
            loss_train = y_hat - Y_rand
            #Calculcate cost aka MSE
            cost_train = float(np.sum(loss_train ** 2))
            #print(f'Loop:{loop+1} => Cost:{cost_train}')
            # MSE list
            MSE_train.append(cost_train) 
            #Calculate Gradient Vector
            gradMSEw = 2*(np.dot(X_rand,loss_train)) + alpha*w
            #Update weights using L2 Regularization (Ridge)
            w = w - eta*gradMSEw
        mean_cost = sum(MSE_train)/len(MSE_train)
        #Calculate least cost and update bestWeights
        if(mean_cost <= leastCost):
            leastCost = cost_train
            bestWeights = w
            eta = eta/1.02 #Add learning decay
        if (mean_cost <= acceptableError):
            break
    #Show least Cost and Best Weight
    print(f'Least Cost = {leastCost} and Best Weights = {bestWeights}')
    #Drop features with weight almost zero from both Training data
    dropColumns = []
    min_wt = np.min(bestWeights)
    for wt,pos in zip(bestWeights,range(len(bestWeights))):
        if min_wt == wt[0]:
            dropColumns.append(pos)
            print(f'We should drop this feature at position {pos+1} having weight {wt[0]}')
    print(f'Dropping feature at index positions:{dropColumns}')
    print(f'Dropping feature : {X_train.columns[dropColumns]}')
    #Dropping Features from Training & Test Data
    X_train.drop(X_train.columns[dropColumns],axis = 1, inplace=True)
    X_test.drop(X_test.columns[dropColumns],axis = 1, inplace=True)
    print(f'Remaining features:{X_train.columns}')
    #Now calling Batch GD after performing L2 regularization
    print(f'Calling Stochastic Gradient Descent after L2 regularization ...')
    gradientDescentStochasticNR(X_train,y_train,X_test,y_test,eta,acceptableError,epoch)

#Call Stochastic GD with L2 regularization
gradientDescentStochasticL2(X_train.copy(),y_train.copy(),X_test.copy(),y_test.copy(),0.01,0.1,10)