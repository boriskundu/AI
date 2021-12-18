# ## ***Perceptron to classify between handwritten didgits (0s and 1s)*** ##
# ## ***Author - Boris Kundu*** ##

#Import packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Read image features from MNISTnumImages5000_balanced
X = pd.read_csv('MNISTnumImages5000_balanced.txt',delimiter='\t',header=None)
#Print image feature shape
print(f'\nImage feature shape: {X.shape}')
#Show image feature head
X.head()

#Display image feature Info
X.info()

#Reshape each row value to get a 28x28 image out of it
X_images = X.values.reshape(5000, 28, 28)
#Define totalImages
totalImages = len(X_images)
#Trsnpose X_images to get proper image
for i in range(totalImages):
    X_images[i] = X_images[i].T
#Display image shape
print(f'\nImages Shape:{X_images.shape}')

#Read image labels from MNISTnumImages5000_balanced
Y = pd.read_csv('MNISTnumLabels5000_balanced.txt',header=None)
#Give name to column
Y.columns = ['label']
#Print image label shape
print(f'\nImage label shape: {Y.shape}')
#Show image label head
Y.head()

#Display image label Info
Y.info()

#Merge X and Y
XM = pd.concat([X,Y],axis=1)
#Add bias feature as 1
XM['bias'] = np.ones(len(XM))
#Check head after adding a bias feature and target variable label.
XM.head()

#Split XN into dataframes for labels = 0, 1, 7 and 9 respectively.
X0 = XM[XM['label']==0]
X1 = XM[XM['label']==1]
X7 = XM[XM['label']==7]
X9 = XM[XM['label']==9]
#Drop label before exporting
X0.drop(columns='label',inplace=True)
X1.drop(columns='label',inplace=True)
X7.drop(columns='label',inplace=True)
X9.drop(columns='label',inplace=True)

#Export CSV for images of 0, 1, 7 and 9 digits
X0.to_csv(r'C:\Users\boris\Python\IS\3\Images_Zero.csv', header = False, index = False)
X1.to_csv(r'C:\Users\boris\Python\IS\3\Images_One.csv', header = False, index = False)
X7.to_csv(r'C:\Users\boris\Python\IS\3\Images_Seven.csv', header = False, index = False)
X9.to_csv(r'C:\Users\boris\Python\IS\3\Images_Nine.csv', header = False, index = False)

#Read all points each from 0 and 1 files
X0 = pd.read_csv('Images_Zero.csv',header=None)
X1 = pd.read_csv('Images_One.csv',header=None)
#Add label to keep track of 0 and 1 digits
X0['label'] = 0
X1['label'] = 1
#Take first 400 from each set for training 
X_0_train = X0[:400]
X_1_train = X1[:400]
#Take rmeaining 100 from each set for testing
X_0_test = X0[400:]
X_1_test = X1[400:]

#Create training set from 0s and 1s
X_train = pd.concat([X_0_train,X_1_train])
#Shuffle it randomly
X_train = X_train.sample(frac = 1)
#Create testing set from 0s and 1s
X_test = pd.concat([X_0_test,X_1_test])
#Shuffle it randomly
X_test = X_test.sample(frac = 1)
#Export training and testing data into files
X_train.drop(['label'],inplace=False,axis=1).to_csv(r'C:\Users\boris\Python\IS\2\Training_Set.csv', header = False, index = False)
X_test.drop(['label'],inplace=False,axis=1).to_csv(r'C:\Users\boris\Python\IS\2\Testing_Set.csv', header = False, index = False)

#Read all points each from 7 and 9 files
X7 = pd.read_csv('Images_Seven.csv',header=None)
X9 = pd.read_csv('Images_Nine.csv',header=None)
#Add label to keep track of 7 and 9 digits
X7['label'] = 7
X9['label'] = 9
#Take first 100 from each set for challenge set 
X7_challenge = X7[:100]
X9_challenge = X9[:100]
#Create challenge set from 7s and 1s
X_challenge = pd.concat([X7_challenge,X9_challenge])
#Shuffle it randomly
X_challenge = X_challenge.sample(frac = 1)
#Export challenege data into file
X_challenge.drop(['label'],inplace=False,axis=1).to_csv(r'C:\Users\boris\Python\IS\2\Challenge_Set.csv', header = False, index = False)

#Function to run train simulation
def trainSimulation(X_train,Y_train,X_test,Y_test,epoch,eta,initialWeights):
    #Wrong predictions
    testFrac = 0
    trainFrac = 0
    #Save training error for epochs
    trainError = []
    #Save testing error for epochs
    testError = []
    #Weights to return
    weights = []
    print(f'Epochs:{epoch} and Eta:{eta}')
    #Number of features aka pixels
    n = len(X_train[0])
    print(f'Total Features:{n}')
    #Initialize weights
    w = initialWeights
    #Total Train Images
    m = len(X_train)
    #Total Test Images
    l = len(X_test)
    print(f'Total Train Images:{m}')
    print(f'Total Test Images:{l}')
    for i in range(epoch):
        #Correctly classified
        correctPredictions = 0
        #Append weights
        weights.append(w)
        testFrac = 0
        trainFrac = 0
        #Predict on Test Set using current weights
        print(f'*** Running perceptron train simulation for epoch {i+1} ***')
        for k in range(l):
            s_t = 0
            y_test = 0
            for p in range(n):
                s_t = s_t + w[p]*X_test[k][p]
            if s_t > 0:
                y_test = 1
            else:
                y_test = 0
            if Y_test[k] != y_test:
                testFrac = testFrac + 1
        testError.append(testFrac/l)
        for img in range(m):
            st = 0
            loss = 0
            yt = 0
            for j in range(n):
                st = st + w[j]*X_train[img][j]
            if st > 0:
                yt = 1
            else:
                yt = 0
            #Check loss/error
            loss = Y_train[img] - yt
            if loss == 0:
                correctPredictions = correctPredictions + 1
            else:
                trainFrac = trainFrac + 1
            #Update weight
            w = w + eta*loss*X_train[img]
        trainError.append(trainFrac/m)
        print(f'Correct Train Predictions:{correctPredictions}')
        print(f'Correct Test Predictions:{l-testFrac}')
        print(f'Train Error Fraction:{trainError[-1]}')
        print(f'Test Error Fraction:{testError[-1]}')
        #Exit on perfect classficiation for all images
        if correctPredictions == m:
            print('Stop Training - Perfect Predictions')
            return (weights,trainError,testError)
    print('Stop Training - Timeout')
    return (weights,trainError,testError)

#For Train Simulation
simul_weights = []
X_train = X_train.reset_index(drop=True)
Y_train = X_train['label']
X_train_mat = X_train.drop(['label'],inplace=False,axis=1).to_numpy()
Y_train_mat = Y_train.to_numpy()
epochs = 50 #Numbr of simulations
eta = 0.005 #Learning rate
#Check stopping condition
all_good = 0
#Get random weights
initialWeights = np.random.uniform(0, 0.5, len(X_train_mat[0]))
#For Test Simulation
X_test = X_test.reset_index(drop=True)
Y_test = X_test['label']
X_test_mat = X_test.drop(['label'],inplace=False,axis=1).to_numpy()
Y_test_mat = Y_test.to_numpy()
#Run trainSimulation
(simul_weights,trainErrorFraction,testErrorFraction) = trainSimulation(X_train_mat.copy(),Y_train_mat.copy(),X_test_mat.copy(),Y_test_mat.copy(),epochs,eta,initialWeights)

#Function to plot error fraction for Train & Test
def plotErrorFraction(trainEF,testEF):
    
    E = [i+1 for i in range(len(trainEF))]
    fig,axes = plt.subplots(figsize=(10,5),num='Error Fraction Plot')
    
    axes.set_title('Figure 1 Error Fraction vs Epoch')
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Error Fraction')

    axes.plot(E,testEF,label = 'Testing Set')
    axes.plot(E,trainEF,label = 'Training Set')
    
    axes.legend()
    fig.show()

#Display Error Fraction
plotErrorFraction(trainErrorFraction,testErrorFraction)

#Initialize final weight
final_weight = initialWeights
#Decide final weight
final_weight = simul_weights[-1] #Timeout or Perfect predictions

#Function to run test simulations
def testSimulation(X_test,Y_test,final_weight,weight_type):
    #True Positives
    t_p = 0
    #True Negatives
    t_n = 0
    #False Positives
    f_p = 0
    #False Negatives
    f_n = 0
    #Total Test Images
    m = len(X_test)
    #Total Features aka Pixels
    n = len(X_test[0])
    #Predicted Test Digits
    Y_predicted = []
    print(f'*** Running test simulation with {weight_type} weights. ***')
    for i in range(m):#Image loop
        st = 0
        for j in range(n):#Pixel loop
            st = st + final_weight[j]*X_test[i][j]
        if (st > 0):
            Y_predicted.append(1)
        else:
            Y_predicted.append(0)
        if (Y_predicted[-1] == 1 and Y_test[i] == 1): # 1s as 1s
            t_p = t_p + 1
        elif (Y_predicted[-1] == 0 and Y_test[i] == 0): # 0s as 0s
            t_n = t_n + 1
        elif (Y_predicted[-1] == 1 and Y_test[i] == 0): # 0s as 1s
            f_p = f_p + 1
        elif (Y_predicted[-1] == 0 and Y_test[i] == 1): # 1s as 0s
            f_n = f_n + 1
    return (t_p,t_n,f_p,f_n)


#Define metrics
tpi = 0
tni = 0
fpi = 0
fni = 0
#Gets metrics with initial weights
weight_type = 'initial'
(tpi,tni,fpi,fni) = testSimulation(X_test_mat.copy(),Y_test_mat.copy(),initialWeights,weight_type)
tpf = 0
tnf = 0
fpf = 0
fnf = 0
#Gets metrics with optimal weights
weight_type = 'optimal'
(tpf,tnf,fpf,fnf) = testSimulation(X_test_mat.copy(),Y_test_mat.copy(),final_weight,weight_type)

#Function to calculate precision, recall and F1 score
def getPerformanceMetrics(tp,tn,fp,fn):
    p = 0
    r = 0
    f = 0
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    f = 2*p*r/(p+r)
    return(p,r,f)


#Get performance metrics
pri = 0
rei = 0
fsi = 0
(pri,rei,fsi) = getPerformanceMetrics(tpi,tni,fpi,fni)
#Get performance metrics
prf = 0
ref = 0
fsf = 0
(prf,ref,fsf) = getPerformanceMetrics(tpf,tnf,fpf,fnf)

#Function to plot precision, recall and F1 score with initial and optimal weights
def plotMetrics(pri,rei,fsi,prf,ref,fsf):
    plt.figure(num = 'Performance Metrics')
    #Prepare data frame
    before = [pri,rei,fsi]
    after = [prf,ref,fsf]
    index = ['Precision','Recall','F1 Score']
    X_axis = np.arange(len(index))
    plt.bar(X_axis - 0.2, before, 0.4, label = 'Before Training')
    plt.bar(X_axis + 0.2, after, 0.4, label = 'After Training')
    plt.xticks(X_axis, index)
    plt.title('Figure 2 Performance Comparison')
    plt.xlabel("Metrics")
    plt.ylabel('Score')
    plt.legend()
    plt.show()

#Plot Performance Metrics
plotMetrics(pri,rei,fsi,prf,ref,fsf)

#Function to run challenege simulation
def challengeSimulation(X_challenge,Y_challenge,final_weight):
    #7s as 1s
    q17 = 0
    #7s as 0s
    q07 = 0
    #9s as 1s
    q19 = 0
    #9sas 0s
    q09 = 0
    #Total Test Images
    m = len(X_challenge)
    #Total Features aka Pixels
    n = len(X_challenge[0])
    #Predicted Challenege Digits
    Y_predicted = []
    print(f'*** Running challenge simulation with optimal weights ***')
    #Initialize to zero
    for i in range(m):#Image loop
        st = 0
        for j in range(n):#Pixel loop
            st = st + final_weight[j]*X_challenge[i][j]
        if (st > 0):
            Y_predicted.append(1)
        else:
            Y_predicted.append(0)
        if (Y_predicted[-1] == 1 and Y_challenge[i] == 7): # 7s as 1s
            q17 = q17 + 1
        elif (Y_predicted[-1] == 0 and Y_challenge[i] == 7): # 7s as 0s
            q07 = q07 + 1
        elif (Y_predicted[-1] == 1 and Y_challenge[i] == 9): # 9s as 1s
            q19 = q19 + 1
        elif (Y_predicted[-1] == 0 and Y_challenge[i] == 9): # 9s as 0s
            q09 = q09 + 1
    return (q17,q07,q19,q09)

#Run Challenge Simulation
X_challenge = X_challenge.reset_index(drop=True)
Y_challenge = X_challenge['label']
X_challenge_mat = X_challenge.drop(['label'],inplace=False,axis=1).to_numpy()
Y_challenge_mat = Y_challenge.to_numpy()
q_19 = 0
q_09 = 0
q_17 = 0
q_07 = 0
#Get challenge stats
(q_17,q_07,q_19,q_09) = challengeSimulation(X_challenge_mat.copy(),Y_challenge_mat.copy(),final_weight)
# Creates 2x2
w, h = 2, 2
challenege_matrix = [[0 for x in range(w)] for y in range(h)] 
challenege_matrix[0][0] = q_07
challenege_matrix[0][1] = q_17
challenege_matrix[1][0] = q_09
challenege_matrix[1][1] = q_19

q = pd.DataFrame(challenege_matrix)

#Display challenege matrix
def displayChallengeMatrix(q):
    plt.figure(num = 'Challenge Matrix')
    sns.heatmap(q, annot=True,cbar=False, cmap='coolwarm')
    plt.title('Figure 3 Perceptron Challenge Matrix')
    plt.xticks(ticks=[0.5,1.5],labels=['Q0','Q1'])
    plt.yticks(ticks=[0.5,1.5],labels=['Q7','Q9'])
    plt.show()

#Call Display Challenge Matrix
displayChallengeMatrix(q)

#Function to plot Initial & Final Weight
def showWeightComparions(iwt,fwt):
    fig, axes = plt.subplots(ncols=2, figsize=(8, 4),num='Weight Comparison')
    ax1, ax2 = axes
    
    ax1.set_title('Figure 4a Perceptron Initial Weights')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    ax2.set_title('Figure 4b Perceptron Final Weights')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')                        
                             
    im1 = ax1.matshow(iwt.T)
    im2 = ax2.matshow(fwt.T)

    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)                        
    plt.tight_layout()
    plt.show()

#Reshape Initial & Final Weights
final_weights_reshaped = final_weight[:len(final_weight)-1].reshape(28, 28)
initial_weights_reshaped = initialWeights[:len(initialWeights)-1].reshape(28, 28)

#Display heatmaps
showWeightComparions(initial_weights_reshaped,final_weights_reshaped)