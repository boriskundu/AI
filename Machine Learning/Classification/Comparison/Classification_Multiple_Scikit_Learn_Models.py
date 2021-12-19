"""
# Classification using different models and their comparison.

Dtataset used is **Titanic**: 
> *   Each **record** represent a different **passenger**
> *   Each **feature** respresent a different **characteristics** of the **passenger**
> * **Passengers** are **labeled** as **1 (survived)** and **0 (deceased)**
> * Source: https://www.kaggle.com/c/titanic/data?select=train.csv
> * Please **rename the above file to Titanic.csv** before running code.

# Problem Statement
> * Analyze Titanic dataset to identify relevant features.
> * **Input** features include **PassengerId,	Pclass,	Name,	Sex,	Age,	SibSp,	Parch,	Ticket,	Fare,	Cabin and	Embarked**
> * **Output** feature is **Survived**
> * Train multiple classification models on the training dataset to classify which passengers survived and which of them couldn't.
> * Evaluate performance and compare the different models on the testing dataset.

## Load Dataset
Import **packages**, **read** and **check** **data**.
"""

#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

#Read CSV
data = pd.read_csv('Titanic.csv')

#Check head
print(f'Head\n:{data.head()}')

#Check data structure
print(f'Info\n:{data.info()}')

"""## Pre-processing
Sequence of steps performed to **prepare** the **data** for modeling.

#### Check for missing data
> * Find all **features** with **null** **values**.
"""

#Check for null values in any feature using heatmap
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='plasma').set(title='Check for null values')

#Check average age by passenger class
data.groupby("Pclass").mean()['Age']

"""#### Clean data
> * **Drop** **feature** **Cabin** as it has **too many null values** that cannot be replaced. 
> * **Update** **Age** **feature** as it has **few null values** with **appropriate** **replacements**.
> * **Drop** **features** with **irrelevant** **information** such as **PassengerId** etc.
"""

#Function to update missing passenger age values with mean age by passenger class
# Pclass 1 = 38 yrs, Pclass 2 = 30 yrs and Pclass 3 = 25 yrs
def getMeanAge(parameters):
    age = parameters[0]
    pClass = parameters[1]
    if pd.isnull(age):
        if pClass == 1:
            return 38
        elif pClass == 2:
            return 30
        else:
            return 25
    else:
        return age

#Update missing passenger age.
data['Age'] = data[['Age','Pclass']].apply(getMeanAge,axis=1)

#Drop feature Cabin as it has too many nulls
data.drop('Cabin',axis=1,inplace=True)

#Check for any missing data again
plt.figure()
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='plasma').set(title='Check again for missing data')

#Drop irrelevant ID features like PassengerId, Ticket
data.drop('PassengerId',axis=1,inplace=True)
data.drop('Ticket',axis=1,inplace=True)

#Check head again
print(f'Check Head:\n{data.head()}')

"""### Feature Engineering
> * Create new feature called **Title** from feature **Name**.
> * **Drop** feature **Name** as it is not relevant.
"""

#Function to extract title (name prefix) from name
def getTitleFromName(parameters):
  name = parameters[0]
  first = ', '
  last = '.'
  start = name.index(first) + len(first)
  end = name.index(last,start)
  return name[start:end]

#Create new feature Title
data['Title'] = data[['Name']].apply(getTitleFromName,axis=1)

#Check unique Titles
data['Title'].unique()

#Drop feature Name
data.drop('Name',axis=1,inplace=True)

#Chec data head again
print(f'Check head again:\n{data.head()}')

"""### Encoding
> * **Convert** **categorical** **features** values into **numeric** ones.
"""

#Encode non-numeric features
le = LabelEncoder()
categorical_columns = ['Sex','Embarked','Title']
for column in categorical_columns:
    data[column]=le.fit_transform(data[column])

#Check head again
print(f'Check head again:\n{data.head()}')

"""### Mark features and label

"""

features = data.drop(columns=['Survived'])
label = data['Survived']

#Check features
print(f'Feature Info:\n{features.info()}')

#Check label
print(f'Label:\n{label}')

"""**0** means passenger **deceased** and **1** means passenger **survived**

## Visualize Data
Check different types of illustrations to understand the dataset better.

#### Check data balance
> * From the below plot we can see **data is not balanced**. 
> * **More** number of **passengers** who **died** (around 550) than the ones who survived (around 350).
> * **More** than half the **passengers** who **survived** were **female** (around 220).
> * **Most** of the **passengeres** who **died** were **male** (around 470).
> * **Most** of the passengers who **survived** belong to **frst and middle class**.
> * **Most** of the **passengers** on board were **young**.
> * **Most** of the **passengers** paid lower fares and belonged to the **lower** **class**.
"""

#Check how many passengers survived or deceased
plt.figure()
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=data).set(title='Check counts by class')

#Check how many male and female passengers survived or deceased
#0 is female and 1 is male
plt.figure()
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=data,hue='Sex').set(title='Check counts by sex')

#Check how many passengers survived or deceased by Passenger Class
#1 is upper, 2 is middle and 3 is lower class
plt.figure()
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=data,hue='Pclass').set(title='Check counts by passenger class')

#Check age distribution of passengers
plt.figure()
sns.histplot(data['Age'],kde=False,color='blue',bins=25).set(title='Age histogram')

#Check fare distribution for passengers
plt.figure()
sns.histplot(data['Fare'],kde=False,color='red',bins=50).set(title='Fare histogram')

"""#### Pairplot
> * Features **Sex** and **PClass** seem to be of more help in distinguishing between deceased and surviving passengers.
"""

#Plot each pair of features
sns.pairplot(data=data,hue='Survived').set(title='Plot all features against each other')

"""#### Check Feature Correlation
> * High negative corelation between '**Fare**' and '**PClass**' which is expected.
> * High positive corelation between '**Parch**' (Parent/Child) and '**SibSp**' (Sibling/Spouse) which is expected.
"""

#Check heatmap. 0 means no corelation.
plt.figure()
sns.heatmap(features.corr(),annot=True, linewidths=.5).set(title='Check feature corelation')

"""### Normalize Data
Standardizing data as some features like '**Fare**' and '**Age**' etc. have **very different rang**es and can add to inefficiency.
"""

#Standardize data
ss = StandardScaler()
scaled_features = ss.fit_transform(features)

#Check scaled features
print(f'Scaled Features:\n{scaled_features}')

#Create scaled feature data frame
scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)

#Check head of normalized data frame
print(f'Scaled Features DF:\n{scaled_features_df.head()}')

"""### Data Statistics
Check **mean**, **quartiles**, **counts** and **standard deviation** for different features.
"""

#Describe data
print(f'Stats:\n{data.describe()}')

"""### Split Data
Split dataset into Train and Test datasets with **70% as Train** size and **30% as Test** size.
"""

#Split data
X_train, X_test, Y_train, Y_test = train_test_split(scaled_features_df, label, test_size = 0.3,random_state = 101)

"""## Model Selection
**Create**, **fit** and **predict** using various classification models to identify the best among them.
"""

#Function to display model performance
def displayResults(model_name,model_predictions):
  print(f'{model_name} Result:')
  accuracy = accuracy_score(Y_test,model_predictions)
  f1 = f1_score(Y_test,model_predictions,average='weighted')
  precision = precision_score(Y_test,model_predictions,average='weighted')
  recall = recall_score(Y_test,model_predictions,average='weighted')
  print('Accuracy: ',accuracy)
  print('F1 Score: ',f1)
  print('Precision: ',precision)
  print('Recall: ', recall)
  return(accuracy,f1,precision,recall)

#Function to plot confusion matrix
def plotConfusionMatrix(model_name,model_predictions,model):
  print(f'{model_name} Confusion Matrix:')
  cm = confusion_matrix(Y_test, model_predictions, labels = model.classes_)
  disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = model.classes_)
  disp.plot()
  plt.grid(False)
  plt.show()

#Function to calculate True Positives, True Negatives, False Positives and False Negatives 
def getPerfMeasure(expected, predicted):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(predicted)): 
        if expected[i] == predicted[i] == 1:
           TP += 1
        if predicted[i] == 1 and expected[i] != predicted[i]:
           FP += 1
        if expected[i] == predicted[i] == 0:
           TN += 1
        if predicted[i] == 0 and expected[i] != predicted[i]:
           FN += 1
    return (TP, FP, TN, FN)

"""### Linear SVC Model"""

#Apply Linear SVC
svc_model = LinearSVC(max_iter=5000)
svc_model.fit(X_train, Y_train)
svc_model_prediction = svc_model.predict(X_test)

#Display Results
(svc_accuracy,svc_f1,svc_precision,svc_recall) = displayResults('Linear SVC',svc_model_prediction)

#Plot confusion matrix
plotConfusionMatrix('Linear SVC',svc_model_prediction,svc_model)

#Get True Positives, True Negatives, False Positives and False Negatives
(TP_svc, FP_svc, TN_svc, FN_svc) = getPerfMeasure(Y_train.to_numpy(),svc_model_prediction)

"""### Logistic Regression Model"""

#Create and fit Logistic Regression Model
LGR_model = LogisticRegression(max_iter=5000)
LGR_model.fit(X_train,Y_train)
LGR_model_prediction = LGR_model.predict(X_test)

#Display Results
(lgr_accuracy,lgr_f1,lgr_precision,lgr_recall) = displayResults('Logistic Regression',LGR_model_prediction)

#Plot confusion matrix
plotConfusionMatrix('Logistic Regression',LGR_model_prediction,LGR_model)

#Get True Positives, True Negatives, False Positives and False Negatives
(TP_lgr, FP_lgr, TN_lgr, FN_lgr) = getPerfMeasure(Y_train.to_numpy(),LGR_model_prediction)

"""### K-Nearest Neighbours Model"""

#Create and fit KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,Y_train)
#Make predictions
knn_prediction = knn.predict(X_test)

#Display Results
(knn_accuracy,knn_f1,knn_precision,knn_recall) = displayResults('K-Nearest Neighbours',knn_prediction)

#Plot confusion matrix
plotConfusionMatrix('K-Nearest Neighbours',knn_prediction,knn)

#Get True Positives, True Negatives, False Positives and False Negatives
(TP_knn, FP_knn, TN_knn, FN_knn) = getPerfMeasure(Y_train.to_numpy(),knn_prediction)

"""### Perceptron Model"""

#Create multi-perceptron model
MLP = MLPClassifier(activation='logistic',solver='lbfgs',hidden_layer_sizes=(1,),max_iter=5000)
#Solver=lbfgs
MLP.fit(X_train, Y_train)
MLP_Predicted = MLP.predict(X_test)

#Display Results
(mlp_accuracy,mlp_f1,mlp_precision,mlp_recall) = displayResults('Multi-Perceptron',MLP_Predicted)

#Plot confusion matrix
plotConfusionMatrix('Perceptron',MLP_Predicted,MLP)

#Get True Positives, True Negatives, False Positives and False Negatives
(TP_mlp, FP_mlp, TN_mlp, FN_mlp) = getPerfMeasure(Y_train.to_numpy(),MLP_Predicted)

"""### Model Comparison"""

#Function to plot accuracy, precision, recall and F1 score of different models for comparison
def plotMetrics(svc_accuracy,svc_f1,svc_precision,svc_recall,lgr_accuracy,lgr_f1,lgr_precision,lgr_recall,knn_accuracy,knn_f1,knn_precision,knn_recall,mlp_accuracy,mlp_f1,mlp_precision,mlp_recall):
    plt.figure(num = 'Performance Metrics')
    #Prepare data frame
    svc = [svc_accuracy,svc_f1,svc_precision,svc_recall]
    lgr = [lgr_accuracy,lgr_f1,lgr_precision,lgr_recall]
    knn = [knn_accuracy,knn_f1,knn_precision,knn_recall]
    mlp = [mlp_accuracy,mlp_f1,mlp_precision,mlp_recall]
    index = ['Accuracy','F1 Score','Precision','Recall']
    X_axis = np.arange(len(index))
    plt.bar(X_axis + 0.1, svc, 0.2, label = 'Linear SVC')
    plt.bar(X_axis - 0.1, lgr, 0.2, label = 'Logistic Regression')
    plt.bar(X_axis + 0.2, knn, 0.1, label = 'K-Nearest Neighbours')
    plt.bar(X_axis - 0.2, mlp, 0.1, label = 'Perceptron')
    plt.xticks(X_axis, index)
    plt.title('Performance Comparison')
    plt.xlabel("Metrics")
    plt.ylabel('Score')
    plt.legend()
    plt.show()

#Function to plot TP,FEP,TN and FN for different models.
def plotPerfMetrics(TP_svc, FP_svc, TN_svc, FN_svc,TP_lgr, FP_lgr, TN_lgr, FN_lgr,TP_knn, FP_knn, TN_knn, FN_knn,TP_mlp, FP_mlp, TN_mlp, FN_mlp):
    plt.figure(num = 'Model Statistics')
    #Prepare data frame
    svc = [TP_svc,FP_svc,TN_svc,FN_svc]
    lgr = [TP_lgr,FP_lgr,TN_lgr,FN_lgr]
    knn = [TP_knn, FP_knn, TN_knn, FN_knn]
    mlp = [TP_mlp, FP_mlp, TN_mlp, FN_mlp]
    index = ['True Positives','False Positives','True Negatives','False Negatives']
    X_axis = np.arange(len(index))
    plt.bar(X_axis + 0.1, svc, 0.2, label = 'Linear SVC')
    plt.bar(X_axis - 0.1, lgr, 0.2, label = 'Logistic Regression')
    plt.bar(X_axis + 0.2, knn, 0.1, label = 'K-Nearest Neighbours')
    plt.bar(X_axis - 0.2, mlp, 0.1, label = 'Perceptron')
    plt.xticks(X_axis, index)
    plt.title('Model Statistics Comparison')
    plt.xlabel("Test Statistics")
    plt.ylabel('Score')
    plt.legend()
    plt.show()

#Functio to plot the feature weights from different models
def plotFeatureWeights(svc_weights,lgr_weights,mlp_weights):
    plt.figure(num = 'Feature Weights')
    index = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title']
    X_axis = np.arange(len(index))
    plt.bar(X_axis + 0.1, svc_weights, 0.2, label = 'Linear SVC')
    plt.bar(X_axis - 0.1, lgr_weights, 0.2, label = 'Logistic Regression')
    plt.bar(X_axis - 0.2, mlp_weights, 0.1, label = 'Perceptron')
    plt.xticks(X_axis, index)
    plt.title('Feature Weights Compariosn')
    plt.xlabel("Features")
    plt.ylabel('Weight')
    plt.legend()
    plt.show()

"""#### Performance
> * Performance metrics such as **Accuracy**, **Precision**, **Recall** and **F1** **Score** were **similar** among the classifier models. 
> * However, **KNN model with 5 neighbours seems to have the best performance** and is closely followed by the Perceptron, Logistic Regression and Linear SVC models.

"""

#Compare models
plotMetrics(svc_accuracy,svc_f1,svc_precision,svc_recall,lgr_accuracy,lgr_f1,lgr_precision,lgr_recall,knn_accuracy,knn_f1,knn_precision,knn_recall,mlp_accuracy,mlp_f1,mlp_precision,mlp_recall)

"""#### Test Statistics
> * **KNN** model had **best** **predictions** for **passengers** who **survived** (True Positives).
> * **Perceptron** model had **best** **predictions** for **deceased** passengers (True Negatives).
"""

#Compare Test Statistics
plotPerfMetrics(TP_svc, FP_svc, TN_svc, FN_svc,TP_lgr, FP_lgr, TN_lgr, FN_lgr,TP_knn, FP_knn, TN_knn, FN_knn,TP_mlp, FP_mlp, TN_mlp, FN_mlp)

"""#### Feature Weights
> * Feature **weights** indicate that **Sex** and **PClass** had the **most say in the survival of a passenger**.
> * Features **Parch** (Parent/Child) and **Title** seem to have the least bearing on the survival of a passenger. These two features **can be removed**.
> * Features such as **Fare** can be represented by **PClass** and can be dropped as well.
"""

#Plot feature weights
plotFeatureWeights(svc_model.coef_[0],LGR_model.coef_[0],[weights[0] for weights in MLP.coefs_[0]])

"""## Intution
> * **Exploratory data analysis** revealed the need for **cleaning**, **normalizing** (standardizing/scaling), **dropping irrelevant features** and **encoding** **categorical** **features** in data.
> * **Data visualization** helped in uderstanding the **balance, relevance** and **correlation among features**.
> * Performance of different classifiers were similar in nature with some marginally better than others.
> * **KNN** predicted the **best** when it came to **surviving passengers (True Positives)** while **Perceptron** gave the **best** results for **deceased passengers (True Negatives)**
> * **Different models assigned different weights to the same set of features**. However, the **general opinion was that Sex and PClass seem to be the most relevant features for determining the survival of a  passenger**.
> * **Dropping** **features** with the **least weights** like **Title** and '**Parch** for specific models may improve their efficiency.
> * **Dropping** features like **Fare** that can be **represented** **by** other feature say **PClass** as they are **corelated** should not alter the performance.
> * **Males** belonging to the **lower class** **suffered** the **most**.
"""

#Check count by group 
group = data.groupby(['Survived','Sex','Pclass']  )
print(f'Counts by Group:\n{group.count()}')