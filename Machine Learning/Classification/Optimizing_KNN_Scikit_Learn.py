# Optimizing KNN
# Author: BORIS KUNDU

#Import packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from sklearn.preprocessing import LabelEncoder

#Read IRIS.csv
data = pd.read_csv('IRIS.csv')

# # Rename columns & encode labels
#Rename columns
data = data.rename(columns={'sepal_length':'X1', 'sepal_width':'X2', 'petal_length':'X3', 'petal_width':'X4','species':'L'})

#Now let's form a label Encoder model
le = LabelEncoder()
#Now we use feed the label column to the model
le.fit(data['L'])
#Model will go through column and find the unique labels (Number of classes that are there)
#Following line will print the labels found in the column
print(f'Classes:{list(le.classes_)}')

#Lets replace these numbers with the labels in data
data['L']=le.transform(data['L'])

# # Standardize Data
#Normalize data and check few rows
scaler = StandardScaler()
scaler.fit(data.drop(columns='L'))
data_scaled = scaler.transform(data.drop(columns='L'))
print(f'Few rows below:\n {data_scaled[:5]} \n Shape: {data_scaled.shape}')

# # Split Data
# 30% Training Data
X_training, X_testing, Y_training, Y_testing = train_test_split(data_scaled, data['L'], test_size=0.3)

# # Perform KNN for different values of K & display graph
#Function to display performance graph
def display(k,f1,accuracy,recall,precision):
    neighbours = [n+1 for n in range(k)]
    #Plot Evaluation metrics 
    fig,axes = plt.subplots(figsize=(10,5),num='Model Evaluation')
    #Define labels to display
    axes.set_title('Performance')
    axes.set_xlabel("Neighbours")
    axes.set_ylabel("Score")
    axes.set_ylim([0.8,1.1])
    axes.set_xlabel("Neighbours")
    axes.set_xticks(neighbours)
    #Plot F1,Accuracy, Precision & Recall
    axes.plot(neighbours,accuracy,label = 'Accuracy')
    axes.plot(neighbours,precision,label = 'Precision')
    axes.plot(neighbours,recall,label = 'Recall')
    axes.plot(neighbours,f1,label = 'F1 Score')
    axes.legend()
    plt.show()

#Function to create and fit KNN model
def KNN(k):
    #Define Lists to store results
    f1 = []
    accuracy = []
    recall = []
    precision = []
    for i in range(k):
        knn = KNeighborsClassifier(n_neighbors=i+1)
        #Get best weights
        knn.fit(X_training,Y_training)
        #Predict classes for Testing dataset
        Y_testing_predicted = knn.predict(X_testing)
        # Calclate model evaluation parameters
        accuracy.append(accuracy_score(Y_testing,Y_testing_predicted))
        f1.append(f1_score(Y_testing,Y_testing_predicted,average='weighted'))
        precision.append(precision_score(Y_testing,Y_testing_predicted,average='weighted'))
        recall.append(recall_score(Y_testing,Y_testing_predicted,average='weighted'))
    #Call display
    display(k,f1,accuracy,recall,precision)

#Call KNN 30 times
KNN(30)