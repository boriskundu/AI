#!/usr/bin/env python
# coding: utf-8

# **Decision Tree (ID3) and Naive Bayes on Iris dataset**
# 
# Authors - BORIS KUNDU

#Import packages
import numpy as np
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import copy
from math import exp
from math import pi
from math import sqrt
from sklearn.model_selection import train_test_split

#Read IRIS data
df = pd.read_csv('iris.csv')

#Perform exploratory data analysis
#Check head
print(f'Head:\n{df.head()}')

#Check data model
print(f'Info:\n{df.info()}')

#Describe data
print(f'Stats:\n{df.describe()}')

#Check pairplot
sns.pairplot(df)

#Combine 'virginica' and 'versicolor' to create a new class 'non-setosa'
def toNonSetosa(species):
    if species != 'setosa':
        return 'non-setosa'
    else:
        return 'setosa'

#Apply transformation
df['species'] = df['species'].apply(lambda x: "non-setosa" if x != "setosa" else "setosa")

#Check tail
print(f'Tail:\n{df.tail()}')

#Tranform categorical feature 'species' to numeric form
#1 means 'setosa' and 0 means 'non-setosa'
#Transform feature sex by changing female to 0 and male to 1
df['species'].replace(['setosa','non-setosa'],[1,0],inplace=True)

#Check head again
print(f'Check Head:\n{df.head()}')

#Check corelation heatmap for numeric features
#1 indicates perfect corelation
plt.figure()
sns.heatmap(df.corr(),annot=True)

#Divide data into 'setosa' and 'non-setosa'
df_setosa = df[df['species']==1] #50
df_non_setosa = df[df['species']==0] #100

#Create 5 smaller datasets with equal ratio of 'setosa' and 'non-setosa' data points
df1 = pd.concat([df_setosa[0:10],df_non_setosa[0:20]],axis=0)
df2 = pd.concat([df_setosa[10:20],df_non_setosa[20:40]],axis=0)
df3 = pd.concat([df_setosa[20:30],df_non_setosa[40:60]],axis=0)
df4 = pd.concat([df_setosa[30:40],df_non_setosa[60:80]],axis=0)
df5 = pd.concat([df_setosa[40:50],df_non_setosa[80:100]],axis=0)

#Define training features in X1,X2,X3,X4,X5
X1 = df1[['sepal_length','sepal_width','petal_length','petal_width']]
X2 = df2[['sepal_length','sepal_width','petal_length','petal_width']]
X3 = df3[['sepal_length','sepal_width','petal_length','petal_width']]
X4 = df4[['sepal_length','sepal_width','petal_length','petal_width']]
X5 = df5[['sepal_length','sepal_width','petal_length','petal_width']]
#Define output target feature in Y1,Y2,Y3,Y4,Y5
Y1 = df1[['species']]
Y2 = df2[['species']]
Y3 = df3[['species']]
Y4 = df4[['species']]
Y5 = df5[['species']]

#Create Train and Test splits for all 5 mini datasets
#Split data randomly into Train and Test with 33% as Test size
#Set 1
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X1, Y1, test_size=0.33, random_state=101)
#Set 2
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, Y2, test_size=0.33, random_state=101)
#Set 3
X_train3, X_test3, Y_train3, Y_test3 = train_test_split(X3, Y3, test_size=0.33, random_state=101)
#Set 4
X_train4, X_test4, Y_train4, Y_test4 = train_test_split(X4, Y4, test_size=0.33, random_state=101)
#Set 5
X_train5, X_test5, Y_train5, Y_test5 = train_test_split(X5, Y5, test_size=0.33, random_state=101)

#Function to calculate accuracy, precision, recall and F1 score
def getPerformanceMetrics(tp,tn,fp,fn):
    a = 0
    p = 0
    r = 0 #Recall = True Positive Rate
    f = 0
    fpr = 0 #False Positve Rate
    a = (tp+tn)/(tp+fp+tn+fn)
    if(tp+fp!=0):#Prevent divide by 0
        p = tp/(tp+fp)
    else:
        p = 0
    if(tp+fn!=0):#Prevent divide by 0
        r = tp/(tp+fn)
    else:
        r = 0
    if(p+r!=0):#Prevent divide by 0
        f = 2*p*r/(p+r)
    else:
        f = 0
    if(tn+fp!=0):#Prevent divide by 0
        fpr = fp/(tn+fp)
    else:
        fpr = 0
    return(a,p,r,f,fpr)

#Function to find average of two numbers
def getMean(x,y):
    average = (x+y)/2
    return average

#Function to calculate mean, standard deviation and length for every training feature
def getTrainFeatureParameters(X_train_full):
    parameters = []
    features = X_train_full.columns
    for feature in features:
        mean = X_train_full[feature].mean()
        std = X_train_full[feature].std()
        if std == 0.0:
            std = 0.000001 #Prevent divide by 0
        length = len(X_train_full[feature])
        parameter = (mean,std,length)
        if feature != 'species':
            parameters.append(parameter)
    return parameters

#Function to find list of mean values of the consecutive elemnts in the number list
def getMeanList(numList):
    length = len(numList) -1
    meanList = [getMean(numList[i],numList[i+1]) for i in range(length)]
    return meanList

#Function to calculate Gaussian PDF
def getProbability(x, mean, std):
    power = (-1/2)*(((x-mean)/std)**2)
    exponent = exp(power)
    if(exponent == 0.0):#Prevent python auto round-off to 0.0 while using math.exp()
        exponent = 0.000001
    pdf = (1/(std*sqrt(2*pi)))*exponent
    return pdf

#Function for calculating entropy
def getEntropy(entity):
    if len(entity) < 2:
        return 0 
    #Calculate count
    count = np.array(entity.value_counts(normalize=True))
    #Calculate entropy
    entropy = -(count * np.log2(count + 1e-6)).sum()
    return entropy

#Function to calculate probabilities and predicting the class for a given row using Naive Bayes
def predictClassNaiveBayes(X_test, Y_test, setosaParams, nonSetosaParams):
    #Convert to matrix
    X_test_matrix = X_test.to_numpy()
    Y_test_matrix = Y_test.to_numpy()
    total_rows = len(X_test_matrix) #data points
    total_columns = len(X_test_matrix[0]) #features
    probSetosa = 0
    probNonSetosa = 0
    Y_predicted = []
    #True Positives
    t_p = 0
    #True Negatives
    t_n = 0
    #False Positives
    f_p = 0
    #False Negatives
    f_n = 0
    for row in range(total_rows):
        probSetosa = 1
        probNonSetosa = 1
        mean_setosa = 0 
        std_setosa = 0
        count_setosa = 0
        mean_non_setosa = 0 
        std_non_setosa = 0
        count_non_setosa = 0
        for column in range(total_columns):
            mean_setosa, std_setosa, count_setosa = setosaParams[column]
            mean_non_setosa, std_non_setosa, count_non_setosa = nonSetosaParams[column]
            probSetosa = probSetosa * getProbability(X_test_matrix[row][column],mean_setosa,std_setosa)
            probNonSetosa = probNonSetosa * getProbability(X_test_matrix[row][column],mean_non_setosa,std_non_setosa)
        #Predict class
        if(probSetosa > probNonSetosa): #It's Setosa!!
            Y_predicted.append(1)
        else: #It's Not Setosa!! --> No Tie Breakers plzzz :D
            Y_predicted.append(0)
        #Calculate loss/error
        if (Y_predicted[-1] == 1 and Y_test_matrix[row][0] == 1): # 1s as 1s
            t_p = t_p + 1
        elif (Y_predicted[-1] == 0 and Y_test_matrix[row][0] == 0): # 0s as 0s
            t_n = t_n + 1
        elif (Y_predicted[-1] == 1 and Y_test_matrix[row][0] == 0): # 0s as 1s
            f_p = f_p + 1
        elif (Y_predicted[-1] == 0 and Y_test_matrix[row][0] == 1): # 1s as 0s
            f_n = f_n + 1
    Y_predicted_df = pd.DataFrame(Y_predicted,columns = ['species'])
    return (Y_predicted_df,t_p,t_n,f_p,f_n)

#Function for calculating information gain
def getInformationGain (entity,target, attribute):
    values = entity[attribute].value_counts(normalize=True)
    entropy_divided = 0
    for v,fr in values.iteritems():
        intermediate = getEntropy(entity[entity[attribute] == v][target])
        entropy_divided += fr * intermediate
    val = getEntropy(entity[target])
    infoGain = val - entropy_divided
    return infoGain

#Add 'species' to filter 'setosa' and 'non-setosa' data points for all 5 sets
X_train1_full = X_train1.join(Y_train1)
X_train2_full = X_train2.join(Y_train2)
X_train3_full = X_train3.join(Y_train3)
X_train4_full = X_train4.join(Y_train4)
X_train5_full = X_train5.join(Y_train5)

X_test1_full = X_test1.join(Y_test1)
X_test2_full = X_test2.join(Y_test2)
X_test3_full = X_test3.join(Y_test3)
X_test4_full = X_test4.join(Y_test4)
X_test5_full = X_test5.join(Y_test5)

# Clas to build the ID3 Decision tree
class Id3TreeBuilder:
    #Initialize Tree Builder
    def __init__(self,entity,aim):
        self.choice = None
        self.entity = entity
        self.aim = aim
        self.attribute_divided = None
        self.child = None
        self.parent = None
    #Create Tree
    def construct(self):
        aim = self.aim
        entity = self.entity
        #Entity list repetative
        if len(entity[aim].unique()) == 1:
            self.choice = entity[aim].unique()[0]
            return None
        #positive scenario
        else:
            highest_ig = 0
            new_entity = pd.DataFrame()
            for attribute in entity.keys():
                #Skip on goal
                if attribute == aim:
                    continue  
                #unqiue values in the attribute
                new_values = entity[attribute].sort_values().unique()
                #mean of the array
                new_values = getMeanList(new_values)
                for patitioner in new_values:
                    name = attribute +" > " + str(patitioner)
                    new_entity[name] = entity[attribute] >  patitioner
            new_entity[aim] = entity[aim]
            # Passing the highest information gain
            for attribute in new_entity.keys():
                #Skip on goal
                if attribute == aim:
                    continue               
                #Get information gain
                info_gain = getInformationGain(new_entity,aim,attribute)
                #Check for highest gain
                if info_gain > highest_ig:
                    highest_ig = info_gain
                    self.attribute_divided = attribute
            self.child = {}
            #Adding child nodes to the ID3 tree
            for value in new_entity[self.attribute_divided].unique():
                index = new_entity[self.attribute_divided] == value
                self.child[value] = Id3TreeBuilder(entity[index],aim)
                self.child[value].construct()
    #Make predictions
    def predict(self,info):
        if self.choice is not None:
            return self.choice
        else:
            #partitioning attributes column
            attribute_divided = self.attribute_divided
            column,value = re.split(" > ",attribute_divided)
            val = float(value)
            child = self.child[(info[column]).astype(float) > val]
            return child.predict(info)




#Define bins
bins = [5,10,15,20]
#Define 'setosa' and 'non-setosa' feature parameters for all 5 sets
#Set 1
featureParametersForSetosa1 = []
featureParametersForNonSetosa1 = []
#Set 2
featureParametersForSetosa2 = []
featureParametersForNonSetosa2 = []
#Set 3
featureParametersForSetosa3 = []
featureParametersForNonSetosa3 = []
#Set 4
featureParametersForSetosa4 = []
featureParametersForNonSetosa4 = []
#Set 5
featureParametersForSetosa5 = []
featureParametersForNonSetosa5 = []
#Model fitting on training data
for b in bins:
    #Set 1
    X_train1_full_c = X_train1_full[0:b].copy()
    featureParametersSetosa1 = getTrainFeatureParameters(X_train1_full_c[X_train1_full_c['species']== 1])
    featureParametersForSetosa1.append(featureParametersSetosa1)
    featureParametersNonSetosa1 = getTrainFeatureParameters(X_train1_full_c[X_train1_full_c['species']== 0])
    featureParametersForNonSetosa1.append(featureParametersNonSetosa1)
    #Set 2
    X_train2_full_c = X_train2_full[0:b].copy()
    featureParametersSetosa2 = getTrainFeatureParameters(X_train2_full_c[X_train2_full_c['species']== 1])
    featureParametersForSetosa2.append(featureParametersSetosa2)
    featureParametersNonSetosa2 = getTrainFeatureParameters(X_train2_full_c[X_train2_full_c['species']== 0])
    featureParametersForNonSetosa2.append(featureParametersNonSetosa2)
    #Set 3
    X_train3_full_c = X_train3_full[0:b].copy()
    featureParametersSetosa3 = getTrainFeatureParameters(X_train3_full_c[X_train3_full_c['species']== 1])
    featureParametersForSetosa3.append(featureParametersSetosa3)
    featureParametersNonSetosa3 = getTrainFeatureParameters(X_train3_full_c[X_train3_full_c['species']== 0])
    featureParametersForNonSetosa3.append(featureParametersNonSetosa3)
    #Set 4
    X_train4_full_c = X_train4_full[0:b].copy()
    featureParametersSetosa4 = getTrainFeatureParameters(X_train4_full_c[X_train4_full_c['species']== 1])
    featureParametersForSetosa4.append(featureParametersSetosa4)
    featureParametersNonSetosa4 = getTrainFeatureParameters(X_train4_full_c[X_train4_full_c['species']== 0])
    featureParametersForNonSetosa4.append(featureParametersNonSetosa4)
    #Set 5
    X_train5_full_c = X_train5_full[0:b].copy()
    featureParametersSetosa5 = getTrainFeatureParameters(X_train5_full_c[X_train5_full_c['species']== 1])
    featureParametersForSetosa5.append(featureParametersSetosa5)
    featureParametersNonSetosa5 = getTrainFeatureParameters(X_train5_full_c[X_train5_full_c['species']== 0])
    featureParametersForNonSetosa5.append(featureParametersNonSetosa5)

#Class for ID3 model object
class Id3:
    def __init__(self):
        self.root = None
    #Create tree and fit model
    def fit (self,entity,aim):
        self.root = Id3TreeBuilder(entity, aim)
        self.root.construct()

#Set 1
Y_predicted1 = []
tp1 = []
tn1 = []
fp1 = []
fn1 = []
fpr1 = []
#Set 2
Y_predicted2 = []
tp2 = []
tn2 = []
fp2 = []
fn2 = []
fpr2 = []
#Set 3
Y_predicted3 = []
tp3 = []
tn3 = []
fp3 = []
fn3 = []
fpr3 = []
#Set 4
Y_predicted4 = []
tp4 = []
tn4 = []
fp4 = []
fn4 = []
fpr4 = []
#Set 5
Y_predicted5 = []
tp5 = []
tn5 = []
fp5 = []
fn5 = []
fpr5 = []

#Get predictions and performance parameters for all 5 sets for all 4 bins
for k in range(len(bins)):
    #Set 1
    (Y_predicted_1,tp_1,tn_1,fp_1,fn_1) = predictClassNaiveBayes(X_test1.copy(),Y_test1.copy(),featureParametersForSetosa1[k],featureParametersForNonSetosa1[k])
    Y_predicted1.append(Y_predicted_1)
    tp1.append(tp_1)
    tn1.append(tn_1)
    fp1.append(fp_1)
    fn1.append(fn_1)
    #Set 2
    (Y_predicted_2,tp_2,tn_2,fp_2,fn_2) = predictClassNaiveBayes(X_test2.copy(),Y_test2.copy(),featureParametersForSetosa2[k],featureParametersForNonSetosa2[k])
    Y_predicted2.append(Y_predicted_2)
    tp2.append(tp_2)
    tn2.append(tn_2)
    fp2.append(fp_2)
    fn2.append(fn_2)
    #Set 3
    (Y_predicted_3,tp_3,tn_3,fp_3,fn_3) = predictClassNaiveBayes(X_test3.copy(),Y_test3.copy(),featureParametersForSetosa3[k],featureParametersForNonSetosa3[k])
    Y_predicted3.append(Y_predicted_3)
    tp3.append(tp_3)
    tn3.append(tn_3)
    fp3.append(fp_3)
    fn3.append(fn_3)
    #Set 4
    (Y_predicted_4,tp_4,tn_4,fp_4,fn_4) = predictClassNaiveBayes(X_test4.copy(),Y_test4.copy(),featureParametersForSetosa4[k],featureParametersForNonSetosa4[k])
    Y_predicted4.append(Y_predicted_4)
    tp4.append(tp_4)
    tn4.append(tn_4)
    fp4.append(fp_4)
    fn4.append(fn_4)
    #Set 5
    (Y_predicted_5,tp_5,tn_5,fp_5,fn_5) = predictClassNaiveBayes(X_test5.copy(),Y_test5.copy(),featureParametersForSetosa5[k],featureParametersForNonSetosa5[k])
    Y_predicted5.append(Y_predicted_5)
    tp5.append(tp_5)
    tn5.append(tn_5)
    fp5.append(fp_5)
    fn5.append(fn_5)

#Get performance metrics
accuracy1 = []
accuracy2 = []
accuracy3 = []
accuracy4 = []
accuracy5 = []

precision1 = []
precision2 = []
precision3 = []
precision4 = []
precision5 = []

recall1 = []
recall2 = []
recall3 = []
recall4 = []
recall5 = []

f11 = [] 
f12 = []
f13 = []
f14 = []
f15 = []

fpr1 = [] 
fpr2 = []
fpr3 = []
fpr4 = []
fpr5 = []

for j in range(len(bins)):
    #Set 1
    (a1,p1,r1,f1,fpr_1) =  getPerformanceMetrics(tp1[j],tn1[j],fp1[j],fn1[j])
    accuracy1.append(a1)
    precision1.append(p1)
    recall1.append(r1)
    f11.append(f1)
    fpr1.append(fpr_1)
    #Set 2
    (a2,p2,r2,f2,fpr_2) =  getPerformanceMetrics(tp2[j],tn2[j],fp2[j],fn2[j])
    accuracy2.append(a2)
    precision2.append(p2)
    recall2.append(r2)
    f12.append(f2)
    fpr2.append(fpr_2)
    #Set 3
    (a3,p3,r3,f3,fpr_3) =  getPerformanceMetrics(tp3[j],tn3[j],fp3[j],fn3[j])
    accuracy3.append(a3)
    precision3.append(p3)
    recall3.append(r3)
    f13.append(f3)
    fpr3.append(fpr_3)
    #Set 4
    (a4,p4,r4,f4,fpr_4) =  getPerformanceMetrics(tp4[j],tn4[j],fp4[j],fn4[j])
    accuracy4.append(a4)
    precision4.append(p4)
    recall4.append(r4)
    f14.append(f4)
    fpr4.append(fpr_4)
    #Set 5
    (a5,p5,r5,f5,fpr_5) =  getPerformanceMetrics(tp5[j],tn5[j],fp5[j],fn5[j])
    accuracy5.append(a5)
    precision5.append(p5)
    recall5.append(r5)
    f15.append(f5)
    fpr5.append(fpr_5)

#Display Accuracy for Naive Bayes
accuracies = []
for i in range(len(bins)):
    accuracies.append([accuracy1[i]*100,accuracy2[i]*100,accuracy3[i]*100,accuracy4[i]*100,accuracy5[i]*100])
index = 0
print(f'*** Naive Bayes Accuracy ***\n')
for b in bins:
    print(f'Bins:{b}')
    print(f'\tNaive Bayes Accuracies (%):{accuracies[index]}')
    print(f'\tMin Acc (%):{min(accuracies[index])}')
    print(f'\tMax Acc (%):{max(accuracies[index])}')
    print(f'\tAvg Acc (%):{sum(accuracies[index])/len(accuracies[index])}')
    index = index + 1

#Function to plot accuracies
def plotAccuracies(bins,accuracies,title_alog):
    plt.figure(figsize=(10,5),num=title_alog)
    
    S = [i+1 for i in range(len(accuracies[0]))]
    
    axes = plt.axes()
    axes.set_title(title_alog+' - Accuracy Plot')
    axes.set_xlabel('Testing Set')
    axes.set_xticks(S)
    axes.set_ylabel('Accuracy (%)')
    
    axes.plot(S,accuracies[0],label = 'Bins = 5',marker='o',markersize=25)
    axes.plot(S,accuracies[1],label = 'Bins = 10',marker='o',markersize=20)
    axes.plot(S,accuracies[2],label = 'Bins = 15',marker='o',markersize=15)
    axes.plot(S,accuracies[3],label = 'Bins = 20',marker='o',markersize=10)
              
    axes.legend()
    plt.show()

#Plot accuracies for different bins from different testing sets
plotAccuracies(bins,accuracies,'Naive Bayes')

#Display F1 Score for Naive Bayes
f1_scores_nb = []
for i in range(len(bins)):
    f1_scores_nb.append([f11[i],f12[i],f13[i],f14[i],f15[i]])

#Function to plot F1 scores
def plotF1(bins,f1_scores,title_alog):
    plt.figure(figsize=(10,5),num=title_alog)
    
    S = [i+1 for i in range(len(f1_scores[0]))]
    axes = plt.axes()
    axes.set_title(title_alog+' - F1 Score Plot')
    axes.set_xlabel('Testing Set')
    axes.set_xticks(S)
    axes.set_ylabel('F1 Score')
    
    axes.plot(S,f1_scores[0],label = 'Bins = 5',marker='o',markersize=25)
    axes.plot(S,f1_scores[1],label = 'Bins = 10',marker='o',markersize=20)
    axes.plot(S,f1_scores[2],label = 'Bins = 15',marker='o',markersize=15)
    axes.plot(S,f1_scores[3],label = 'Bins = 20',marker='o',markersize=10)
              
    axes.legend()
    plt.show()

#Plot F1 scores for different bins from different testing sets
plotF1(bins,f1_scores_nb,'Naive Bayes')

#Display ROC Curve for Naive Bayes
fpr = []
tpr = []
for i in range(len(bins)):
    fpr.append([0.0] + [fpr1[i],fpr2[i],fpr3[i],fpr4[i],fpr5[i]])
    tpr.append([0.0] + [recall1[i],recall2[i],recall3[i],recall4[i],recall5[i]])

#Function to plot F1 scores
def plotROC(tpr,fpr,title_alog):
    plt.figure(figsize=(10,5),num=title_alog)
    
    axes = plt.axes()
    axes.set_title(title_alog+' - ROC Curve Plot')
    axes.set_xlabel('False Positive Rate')
    axes.set_ylabel('True Positive Rate')

    axes.plot(fpr[0],tpr[0],label = 'Bins = 5',marker='o',markersize=25)
    axes.plot(fpr[1],tpr[1],label = 'Bins = 10',marker='o',markersize=20)
    axes.plot(fpr[2],tpr[2],label = 'Bins = 15',marker='o',markersize=15)
    axes.plot(fpr[3],tpr[3],label = 'Bins = 20',marker='o',markersize=10)
             
    axes.legend()
    axes.set_xlim(-0.1,1.1)
    axes.set_ylim(-0.1,1.1)
    plt.show()

#Plot ROC for different bins from different testing sets
plotROC(tpr,fpr,'Naive Bayes')

#Function to calculate performance metrics
def calculateMetrics(pred,test_actual):
    t_p = 0
    f_p = 0
    t_n = 0
    f_n = 0
    n = len(pred)
    actual = test_actual.values
    for i in range(0,n):
        if (pred[i] == 1 and actual[i] == 1): # 1s as 1s
            t_p = t_p + 1
        elif (pred[i] == 0 and actual[i] == 0): # 0s as 0s
            t_n = t_n + 1
        elif (pred[i] == 1 and actual[i] == 0): # 0s as 1s
            f_p = f_p + 1
        elif (pred[i] == 0 and actual[i] == 1): # 1s as 0s
            f_n = f_n + 1
    return (t_p,f_p,t_n,f_n)

y_predict_1 = []
y_predict_2 = []
y_predict_3 = []
y_predict_4 = []
y_predict_5 = []

y_pred_test_1 = pd.DataFrame({'species' : [],'id':[]})
y_pred_test_2 = pd.DataFrame({'species' : [],'id':[]})
y_pred_test_3 = pd.DataFrame({'species' : [],'id':[]})
y_pred_test_4 = pd.DataFrame({'species' : [],'id':[]})
y_pred_test_5 = pd.DataFrame({'species' : [],'id':[]})

# initialize and fit model
model1 = Id3()
model2 = Id3()
model3 = Id3()
model4 = Id3()
model5 = Id3()

tp1_id3 = []
tp2_id3 = []
tp3_id3 = []
tp4_id3 = []
tp5_id3 = []

fp1_id3 = []
fp2_id3 = []
fp3_id3 = []
fp4_id3 = []
fp5_id3 = []

tn1_id3 = []
tn2_id3 = []
tn3_id3 = []
tn4_id3 = []
tn5_id3 = []

fn1_id3 = []
fn2_id3 = []
fn3_id3 = []
fn4_id3 = []
fn5_id3 = []

for b in bins:
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    #Set 1
    model1.fit(X_train1_full[0:b].copy(), 'species')
    for i,row in X_test1_full.copy().iterrows():
        y_pred_test_1= y_pred_test_1.append({'species':model1.root.predict(row),'id':int(i) }, ignore_index=True)
    (tp,fp,tn,fn) = calculateMetrics(y_pred_test_1['species'][-10:].values,Y_test1)
    y_predict_1.append(y_pred_test_1['species'][-10:])
    tp1_id3.append(tp)
    fp1_id3.append(fp)
    tn1_id3.append(tn)
    fn1_id3.append(fn)
    
    #Set 2
    model2.fit(X_train2_full[0:b].copy(),'species')
    for i,row in X_test2_full.copy().iterrows():
        y_pred_test_2= y_pred_test_2.append({'species':model2.root.predict(row),'id':int(i) }, ignore_index=True)
    (tp,fp,tn,fn) = calculateMetrics(y_pred_test_2['species'][-10:].values,Y_test2)
    y_predict_2.append(y_pred_test_2['species'][-10:])
    tp2_id3.append(tp)
    fp2_id3.append(fp)
    tn2_id3.append(tn)
    fn2_id3.append(fn)
    
    #Set 3
    model3.fit(X_train3_full[0:b].copy(), 'species')
    for i,row in X_test3_full.copy().iterrows():
        y_pred_test_3= y_pred_test_3.append({'species':model3.root.predict(row),'id':int(i) }, ignore_index=True)
    (tp,fp,tn,fn) = calculateMetrics(y_pred_test_3['species'][-10:].values,Y_test3)
    y_predict_3.append(y_pred_test_3['species'][-10:])
    tp3_id3.append(tp)
    fp3_id3.append(fp)
    tn3_id3.append(tn)
    fn3_id3.append(fn)
    
    #Set 4
    model4.fit(X_train4_full[0:b].copy(),'species' )
    for i,row in X_test4_full.copy().iterrows():
        y_pred_test_4= y_pred_test_4.append({'species':model4.root.predict(row),'id':int(i) }, ignore_index=True)
    (tp,fp,tn,fn) = calculateMetrics(y_pred_test_4['species'][-10:].values,Y_test4)
    y_predict_4.append(y_pred_test_4['species'][-10:])
    tp4_id3.append(tp)
    fp4_id3.append(fp)
    tn4_id3.append(tn)
    fn4_id3.append(fn)

    #Set 5
    model5.fit(X_train5_full[0:b].copy(), 'species' )
    for i,row in X_test5_full.copy().iterrows():
        y_pred_test_5= y_pred_test_5.append({'species':model5.root.predict(row),'id':int(i) }, ignore_index=True)
    (tp,fp,tn,fn) = calculateMetrics(y_pred_test_5['species'][-10:].values,Y_test5)
    y_predict_5.append(y_pred_test_5['species'][-10:])
    tp5_id3.append(tp)
    fp5_id3.append(fp)
    tn5_id3.append(tn)
    fn5_id3.append(fn)

#Get performance metrics
id3accuracy1 = []
id3accuracy2 = []
id3accuracy3 = []
id3accuracy4 = []
id3accuracy5 = []

id3precision1 = []
id3precision2 = []
id3precision3 = []
id3precision4 = []
id3precision5 = []

id3recall1 = []
id3recall2 = []
id3recall3 = []
id3recall4 = []
id3recall5 = []

id3f11 = [] 
id3f12 = []
id3f13 = []
id3f14 = []
id3f15 = []

id3fpr1 = [] 
id3fpr2 = []
id3fpr3 = []
id3fpr4 = []
id3fpr5 = []

for j in range(len(bins)):
    #Set 1
    (a1,p1,r1,id3f1,id3fpr_1) =  getPerformanceMetrics(tp1_id3[j],tn1_id3[j],fp1_id3[j],fn1_id3[j])
    id3accuracy1.append(a1)
    id3precision1.append(p1)
    id3recall1.append(r1)
    id3f11.append(id3f1)
    id3fpr1.append(id3fpr_1)
    #Set 2
    (a2,p2,r2,f2,id3fpr_2) =  getPerformanceMetrics(tp2_id3[j],tn2_id3[j],fp2_id3[j],fn2_id3[j])
    id3accuracy2.append(a2)
    id3precision2.append(p2)
    id3recall2.append(r2)
    id3f12.append(f2)
    id3fpr2.append(id3fpr_2)
    #Set 3
    (a3,p3,r3,f3,id3fpr_3) =  getPerformanceMetrics(tp3_id3[j],tn3_id3[j],fp3_id3[j],fn3_id3[j])
    id3accuracy3.append(a3)
    id3precision3.append(p3)
    id3recall3.append(r3)
    id3f13.append(f3)
    id3fpr3.append(id3fpr_3)
    #Set 4
    (a4,p4,r4,f4,id3fpr_4) =  getPerformanceMetrics(tp4_id3[j],tn4_id3[j],fp4_id3[j],fn4_id3[j])
    id3accuracy4.append(a4)
    id3precision4.append(p4)
    id3recall4.append(r4)
    id3f14.append(f4)
    id3fpr4.append(id3fpr_4)
    #Set 5
    (a5,p5,r5,f5,id3fpr_5) =  getPerformanceMetrics(tp5_id3[j],tn5_id3[j],fp5_id3[j],fn5_id3[j])
    id3accuracy5.append(a5)
    id3precision5.append(p5)
    id3recall5.append(r5)
    id3f15.append(f5)
    id3fpr5.append(id3fpr_5)

#Display id3 accuracy
id3accuracies = []
for i in range(len(bins)):
    id3accuracies.append([id3accuracy1[i]*100,id3accuracy2[i]*100,id3accuracy3[i]*100,id3accuracy4[i]*100,id3accuracy5[i]*100])
index = 0
print(f'\n*** ID3 Accuracy ***\n')
for b in bins:
    print(f'Bins:{b}')
    print(f'\tID3 accuracies (%):{id3accuracies[index]}')
    print(f'\tMin Acc (%):{min(id3accuracies[index])}')
    print(f'\tMax Acc (%):{max(id3accuracies[index])}')
    print(f'\tAvg Acc (%):{sum(id3accuracies[index])/len(id3accuracies[index])}')
    index = index + 1

#Plot accuracies for different bins from different testing sets
plotAccuracies(bins,id3accuracies,'ID3')

#Display F1 Score for Naive Bayes
f1_scores_id3 = []
for i in range(len(bins)):
    f1_scores_id3.append([id3f11[i],id3f12[i],id3f13[i],id3f14[i],id3f15[i]])

#Plot F1 scores for different bins from different testing sets
plotF1(bins,f1_scores_id3,'ID3')

#Display ROC Curve for ID3
fpr_id3 = []
tpr_id3 = []
for i in range(len(bins)):
    fpr_id3.append([0.0] + [id3fpr1[i],id3fpr2[i],id3fpr3[i],id3fpr4[i],id3fpr5[i]])
    tpr_id3.append([0.0] + [id3recall1[i],id3recall2[i],id3recall3[i],id3recall4[i],id3recall5[i]])

#Plot ROC for different bins from different testing sets
plotROC(tpr_id3,fpr_id3,'ID3')

#Using ID3 test predictions as test truth for Naive Bayes
#Set 1
NBY_predicted1 = []
NBtp1 = []
NBtn1 = []
NBfp1 = []
NBfn1 = []
NBfpr1 = []
#Set 2
NBY_predicted2 = []
NBtp2 = []
NBtn2 = []
NBfp2 = []
NBfn2 = []
NBfpr2 = []
#Set 3
NBY_predicted3 = []
NBtp3 = []
NBtn3 = []
NBfp3 = []
NBfn3 = []
NBfpr3 = []
#Set 4
NBY_predicted4 = []
NBtp4 = []
NBtn4 = []
NBfp4 = []
NBfn4 = []
NBfpr4 = []
#Set 5
NBY_predicted5 = []
NBtp5 = []
NBtn5 = []
NBfp5 = []
NBfn5 = []
NBfpr5 = []

#Get predictions and performance parameters for all 5 sets for all 4 bins
for k in range(4):
    #Set 1
    id3out1 = pd.DataFrame(y_predict_1[k],columns=['species'])
    (NBY_predicted_1,NBtp_1,NBtn_1,NBfp_1,NBfn_1) = predictClassNaiveBayes(X_test1.copy(),id3out1,featureParametersForSetosa1[k],featureParametersForNonSetosa1[k])
    NBY_predicted1.append(NBY_predicted_1)
    NBtp1.append(NBtp_1)
    NBtn1.append(NBtn_1)
    NBfp1.append(NBfp_1)
    NBfn1.append(NBfn_1)
    #Set 2
    id3out2 = pd.DataFrame(y_predict_2[k],columns=['species'])
    (NBY_predicted_2,NBtp_2,NBtn_2,NBfp_2,NBfn_2) = predictClassNaiveBayes(X_test2.copy(),id3out2,featureParametersForSetosa2[k],featureParametersForNonSetosa2[k])
    NBY_predicted2.append(NBY_predicted_2)
    NBtp2.append(NBtp_2)
    NBtn2.append(NBtn_2)
    NBfp2.append(NBfp_2)
    NBfn2.append(NBfn_2)
    #Set 3
    id3out3 = pd.DataFrame(y_predict_3[k],columns=['species'])
    (NBY_predicted_3,NBtp_3,NBtn_3,NBfp_3,NBfn_3) = predictClassNaiveBayes(X_test3.copy(),id3out3,featureParametersForSetosa3[k],featureParametersForNonSetosa3[k])
    NBY_predicted3.append(NBY_predicted_3)
    NBtp3.append(NBtp_3)
    NBtn3.append(NBtn_3)
    NBfp3.append(NBfp_3)
    NBfn3.append(NBfn_3)
    #Set 4
    id3out4 = pd.DataFrame(y_predict_4[k],columns=['species'])
    (NBY_predicted_4,NBtp_4,NBtn_4,NBfp_4,NBfn_4) = predictClassNaiveBayes(X_test4.copy(),id3out4,featureParametersForSetosa4[k],featureParametersForNonSetosa4[k])
    NBY_predicted4.append(NBY_predicted_4)
    NBtp4.append(NBtp_4)
    NBtn4.append(NBtn_4)
    NBfp4.append(NBfp_4)
    NBfn4.append(NBfn_4)
    #Set 5
    id3out5 = pd.DataFrame(y_predict_5[k],columns=['species'])
    (NBY_predicted_5,NBtp_5,NBtn_5,NBfp_5,NBfn_5) = predictClassNaiveBayes(X_test5.copy(),id3out5,featureParametersForSetosa5[k],featureParametersForNonSetosa5[k])
    NBY_predicted5.append(NBY_predicted_5)
    NBtp5.append(NBtp_5)
    NBtn5.append(NBtn_5)
    NBfp5.append(NBfp_5)
    NBfn5.append(NBfn_5)

#Get performance metrics
NBaccuracy1 = []
NBaccuracy2 = []
NBaccuracy3 = []
NBaccuracy4 = []
NBaccuracy5 = []

NBprecision1 = []
NBprecision2 = []
NBprecision3 = []
NBprecision4 = []
NBprecision5 = []

NBrecall1 = []
NBrecall2 = []
NBrecall3 = []
NBrecall4 = []
NBrecall5 = []

NBf11 = [] 
NBf12 = []
NBf13 = []
NBf14 = []
NBf15 = []

NBfpr1 = [] 
NBfpr2 = []
NBfpr3 = []
NBfpr4 = []
NBfpr5 = []

for j in range(len(bins)):
    #Set 1
    (a1,p1,r1,NBf1,NBfpr_1) =  getPerformanceMetrics(NBtp1[j],NBtn1[j],NBfp1[j],NBfn1[j])
    NBaccuracy1.append(a1)
    NBprecision1.append(p1)
    NBrecall1.append(r1)
    NBf11.append(NBf1)
    NBfpr1.append(NBfpr_1)
    #Set 2
    (a2,p2,r2,f2,NBfpr_2) =  getPerformanceMetrics(NBtp2[j],NBtn2[j],NBfp2[j],NBfn2[j])
    NBaccuracy2.append(a2)
    NBprecision2.append(p2)
    NBrecall2.append(r2)
    NBf12.append(f2)
    NBfpr2.append(NBfpr_2)
    #Set 3
    (a3,p3,r3,f3,NBfpr_3) =  getPerformanceMetrics(NBtp3[j],NBtn3[j],NBfp3[j],NBfn3[j])
    NBaccuracy3.append(a3)
    NBprecision3.append(p3)
    NBrecall3.append(r3)
    NBf13.append(f3)
    NBfpr3.append(NBfpr_3)
    #Set 4
    (a4,p4,r4,f4,NBfpr_4) =  getPerformanceMetrics(NBtp4[j],NBtn4[j],NBfp4[j],NBfn4[j])
    NBaccuracy4.append(a4)
    NBprecision4.append(p4)
    NBrecall4.append(r4)
    NBf14.append(f4)
    NBfpr4.append(NBfpr_4)
    #Set 5
    (a5,p5,r5,f5,NBfpr_5) =  getPerformanceMetrics(NBtp5[j],NBtn5[j],NBfp5[j],NBfn5[j])
    NBaccuracy5.append(a5)
    NBprecision5.append(p5)
    NBrecall5.append(r5)
    NBf15.append(f5)
    NBfpr5.append(NBfpr_5)

#Display F1 Score for Naive Bayes Truth
f1_scores_NBT = []
for i in range(len(bins)):
    f1_scores_NBT.append([NBf11[i],NBf12[i],NBf13[i],NBf14[i],NBf15[i]])

#Plot F1 scores for different bins from different testing sets
plotF1(bins,f1_scores_NBT,'Naive Bayes (ID3 test predictions as truth)')

#Function to calculate performance metrics
def calculateMetrics1(pred,test_actual):
    t_p = 0
    f_p = 0
    t_n = 0
    f_n = 0
    n = len(pred)
    pre = pred.to_numpy()
    act = test_actual.to_numpy()
    for i in range(n):
        if (pre[i] == 1 and act[i] == 1): # 1s as 1s
            t_p = t_p + 1
        elif (pre[i] == 0 and act[i] == 0): # 0s as 0s
            t_n = t_n + 1
        elif (pre[i] == 1 and act[i] == 0): # 0s as 1s
            f_p = f_p + 1
        elif (pre[i] == 0 and act[i] == 1): # 1s as 0s
            f_n = f_n + 1
    return (t_p,f_p,t_n,f_n)

#Using Naive Bayes test predictions as test truth for nb
tp1_nb = []
tp2_nb = []
tp3_nb = []
tp4_nb = []
tp5_nb = []

fp1_nb = []
fp2_nb = []
fp3_nb = []
fp4_nb = []
fp5_nb = []

tn1_nb = []
tn2_nb = []
tn3_nb = []
tn4_nb = []
tn5_nb = []

fn1_nb = []
fn2_nb = []
fn3_nb = []
fn4_nb = []
fn5_nb = []

for b in range(0,len(bins)):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    #Set 1
    (tp,fp,tn,fn) = calculateMetrics1(y_predict_1[b],Y_predicted1[b].squeeze())
    tp1_nb.append(tp)
    fp1_nb.append(fp)
    tn1_nb.append(tn)
    fn1_nb.append(fn)
    
    #Set 2
    (tp,fp,tn,fn) = calculateMetrics1(y_predict_2[b],Y_predicted2[b].squeeze())
    tp2_nb.append(tp)
    fp2_nb.append(fp)
    tn2_nb.append(tn)
    fn2_nb.append(fn)
    
    #Set 3
    (tp,fp,tn,fn) = calculateMetrics1(y_predict_3[b],Y_predicted3[b].squeeze())
    tp3_nb.append(tp)
    fp3_nb.append(fp)
    tn3_nb.append(tn)
    fn3_nb.append(fn)
    
    #Set 4
    (tp,fp,tn,fn) = calculateMetrics1(y_predict_4[b],Y_predicted4[b].squeeze())
    tp4_nb.append(tp)
    fp4_nb.append(fp)
    tn4_nb.append(tn)
    fn4_nb.append(fn)

    #Set 5
    (tp,fp,tn,fn) = calculateMetrics1(y_predict_5[b],Y_predicted5[b].squeeze())
    tp5_nb.append(tp)
    fp5_nb.append(fp)
    tn5_nb.append(tn)
    fn5_nb.append(fn)

#Get performance metrics
id3accuracyTruth1 = []
id3accuracyTruth2 = []
id3accuracyTruth3 = []
id3accuracyTruth4 = []
id3accuracyTruth5 = []

id3precisionTruth1 = []
id3precisionTruth2 = []
id3precisionTruth3 = []
id3precisionTruth4 = []
id3precisionTruth5 = []

id3recallTruth1 = []
id3recallTruth2 = []
id3recallTruth3 = []
id3recallTruth4 = []
id3recallTruth5 = []

id3f1Truth1 = [] 
id3f1Truth2 = []
id3f1Truth3 = []
id3f1Truth4 = []
id3f1Truth5 = []

id3fpr1 = [] 
id3fpr2 = []
id3fpr3 = []
id3fpr4 = []
id3fpr5 = []

for j in range(len(bins)):
    #Set 1
    (a1,p1,r1,id3f1Truth,id3fpr_1) =  getPerformanceMetrics(tp1_nb[j],tn1_nb[j],fp1_nb[j],fn1_nb[j])
    id3accuracyTruth1.append(a1)
    id3recallTruth1.append(r1)
    id3f1Truth1.append(id3f1Truth)
    id3fpr1.append(id3fpr_1)
    #Set 2
    (a2,p2,r2,f2,id3fpr_2) =  getPerformanceMetrics(tp2_nb[j],tn2_nb[j],fp2_nb[j],fn2_nb[j])
    id3accuracyTruth2.append(a2)
    id3precisionTruth2.append(p2)
    id3recallTruth2.append(r2)
    id3f1Truth2.append(f2)
    id3fpr2.append(id3fpr_2)
    #Set 3
    (a3,p3,r3,f3,id3fpr_3) =  getPerformanceMetrics(tp3_nb[j],tn3_nb[j],fp3_nb[j],fn3_nb[j])
    id3accuracyTruth3.append(a3)
    id3precisionTruth3.append(p3)
    id3recallTruth3.append(r3)
    id3f1Truth3.append(f3)
    id3fpr3.append(id3fpr_3)
    #Set 4
    (a4,p4,r4,f4,id3fpr_4) =  getPerformanceMetrics(tp4_nb[j],tn4_nb[j],fp4_nb[j],fn4_nb[j])
    id3accuracyTruth4.append(a4)
    id3precisionTruth4.append(p4)
    id3recallTruth4.append(r4)
    id3f1Truth4.append(f4)
    id3fpr4.append(id3fpr_4)
    #Set 5
    (a5,p5,r5,f5,id3fpr_5) =  getPerformanceMetrics(tp5_nb[j],tn5_nb[j],fp5_nb[j],fn5_nb[j])
    id3accuracyTruth5.append(a5)
    id3precisionTruth5.append(p5)
    id3recallTruth5.append(r5)
    id3f1Truth5.append(f5)
    id3fpr5.append(id3fpr_5)

#Display F1 Score for IDS Truth
f1_scores_ID3T = []
for i in range(len(bins)):
    f1_scores_ID3T.append([id3f1Truth1[i],id3f1Truth2[i],id3f1Truth3[i],id3f1Truth4[i],id3f1Truth5[i]])

#Plot F1 scores for different bins from different testing sets
plotF1(bins,f1_scores_ID3T,'ID3 (Naive Bayes test predictions as truth)')