# ## ***A binary threshold neuron to discriminate between two handwritten digits.
# Both post-synaptically and pre-synaptically gated Hebb rules are compared.*** ##
# 
# ## ***Author - BORIS KUNDU*** ##

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
#print(f'\nView some sample images ...')
#Display few images
#plt.figure()
#sns.heatmap(X_images[1000])
#plt.figure()
#sns.heatmap(X_images[2000])
#plt.figure()
#sns.heatmap(X_images[3000])
#plt.figure()
#sns.heatmap(X_images[4000])
#plt.show()

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
X0.to_csv(r'C:\Users\boris\Python\IS\2\Images_Zero.csv', header = False, index = False)
X1.to_csv(r'C:\Users\boris\Python\IS\2\Images_One.csv', header = False, index = False)
X7.to_csv(r'C:\Users\boris\Python\IS\2\Images_Seven.csv', header = False, index = False)
X9.to_csv(r'C:\Users\boris\Python\IS\2\Images_Nine.csv', header = False, index = False)

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
def trainSimulation(X_train,Y_train,epoch,eta,initialWeights):
    #Gather stats
    #stats = {}
    weights = []
    #img_list = []
    print(f'Epochs:{epoch} and Eta:{eta}')
    #print(f'Learning Input Shape:{Y_train.shape}')
    #Number of features aka pixels
    n = len(X_train[0])
    print(f'Total Features:{n}')
    #Initialize weights
    w = initialWeights
    #Total Train Images
    m = len(X_train)
    #print(f'Training Set Shape:{X_train.shape}')
    print(f'Total Images:{m}')
    for i in range(epoch):
        print(f'*** Running postynaptically-gated train simulation for epoch {i+1} ***')
        for img in range(m):
            #print(f'Processing Image {img+1}')
            st = 0
            zt = Y_train[img]
            yt = zt
            prod = 0
            for j in range(n):
                prod = w[j]*X_train[img][j]
                st = st + prod
            #Update weight
            w = w + eta*yt*(X_train[img]-w)
            #Gather stats
            #img_list.append((st,yt))
        weights.append(w)    
        #stats[i]=img_list
    #return (stats,weights)
    return weights

#Run Train Simulation
#simul_stat = {}
simul_weights = []
X_train = X_train.reset_index(drop=True)
Y_train = X_train['label']
X_train_mat = X_train.drop(['label'],inplace=False,axis=1).to_numpy()
Y_train_mat = Y_train.to_numpy()
epochs = 40 #Numbr of simulations
eta = 0.01 #Learning rate
#Get random weights
initialWeights = np.random.uniform(0, 0.5, len(X_train_mat[0]))
#simul_stat,simul_weights = trainSimulation(X_train_mat.copy(),Y_train_mat.copy(),epochs,eta,initialWeights)
simul_weights = trainSimulation(X_train_mat.copy(),Y_train_mat.copy(),epochs,eta,initialWeights)

#Set last used weight as final weight
final_weight = simul_weights[-2]

#Function to run test simulations
def testSimulation(X_test,Y_test,final_weight,theta_max):
    #True Positives
    tp = []
    #True Negatives
    tn = []
    #False Positives
    fp = []
    #False Negatives
    fn = []
    #Total Test Images
    m = len(X_test)
    #Total Features aka Pixels
    n = len(X_test[0])
    #Predicted Test Digits
    Y_predicted = []
    for theta in range(theta_max):#Theta loop
        print(f'*** Running test simulation with theta as {theta} ***')
        #Initialize to zero
        t_p = 0
        t_n = 0
        f_p = 0
        f_n = 0
        for i in range(m):#Image loop
            st = 0
            for j in range(n):#Pixel loop
                prod = final_weight[j]*X_test[i][j]
                st = st + prod
            #print(f'st:{st} and theta:{theta}')
            if (st > theta):
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
        if(t_p+t_n+f_p+f_n == m):
            tp.append(t_p)
            tn.append(t_n)
            fp.append(f_p)
            fn.append(f_n)
    return (tp,tn,fp,fn)

#Run Test Simulation
X_test = X_test.reset_index(drop=True)
Y_test = X_test['label']
X_test_mat = X_test.drop(['label'],inplace=False,axis=1).to_numpy()
Y_test_mat = Y_test.to_numpy()
truePositives = []
trueNegatives = []
falsePositives = []
falseNegatives = []
theta_max = 41# For 0 to 40
#Gets metrics
(truePositives,trueNegatives,falsePositives,falseNegatives) = testSimulation(X_test_mat.copy(),Y_test_mat.copy(),final_weight,theta_max)

#Function to calculate precision, recall and F1 score
def getPerformanceMetrics(tp,tn,fp,fn):
    precision = []
    f1 = []
    recall = []
    p = 0
    r = 0
    f = 0
    for k in range(len(tp)):
        p = tp[k]/(tp[k]+fp[k])
        r = tp[k]/(tp[k]+fn[k])
        f = 2*p*r/(p+r)
        precision.append(p)
        recall.append(r)
        f1.append(f)
    return(precision,recall,f1)

#Get performance metrics
pr = []
re = []
fs = []
(pr,re,fs) = getPerformanceMetrics(truePositives,trueNegatives,falsePositives,falseNegatives)

#Function to plot precision, recall and F1 score
def plotMetrics(precision,recall,f1):
    
    T = [i for i in range(len(precision))]
    fig,axes = plt.subplots(figsize=(10,5), num='Postsynaptic-gated Precision, Recall and F1 Score')
    
    axes.set_title('Figure 1 Postsynaptic-gated Performance Metrics vs Theta')
    axes.set_xlabel('Theta')
    axes.set_ylabel('Metrics')

    axes.plot(T,precision,label = 'Precision')
    axes.plot(T,recall,label = 'Recall')
    axes.plot(T,f1,label = 'F1 Score')
    
    axes.legend()
    fig.show()

#Plot Performance Metrics with respect to Theta
plotMetrics(pr,re,fs)

#Function to plot ROC
def plotROC(tp,fp):
    
    fig,axes = plt.subplots(figsize=(10,5), num='Postsynaptic-gated ROC Curve')
    
    axes.set_title('Figure 2 Postsynaptic-gated ROC Curve')
    axes.set_xlabel('False Positives')
    axes.set_ylabel('True Positives')

    axes.plot(fp,tp,marker='o',markerfacecolor='red')
    fig.show()

#Plot Performance Metrics with respect to Theta
plotROC(truePositives,falsePositives)

#Function to calculate slope of ROC at different points
def findROCSlope(tp,fp):
    slope = []
    for i in range (len(tp)):
        if i == 0:
            slope.append(0)
        else:
            if (fp[i] != fp[i-1]):
                s = (tp[i] - tp[i-1])/(fp[i] - fp[i-1])
            else:
                s = -1 # Divide by 0 is not dfined
            slope.append(s)
    return slope

#Get slope
rocSlope = findROCSlope(truePositives,falsePositives)

#Get optimal theta values
def getOptimalTheta(slope,tp,fp,tn,fn):
    #Optimal Theta
    ot = 0
    ratio = 0
    accuracy = 0
    max_ratio = 0
    max_accuracy = 0
    for s in range(len(slope)):
        if slope[s] > 0 and slope[s] < 1:
            print(f'Candidate Theta:{s} with Slope:{slope[s]}')
            print(f'True Positives:{tp[s]} and False Positives:{fp[s]}')
            ratio = tp[s]/fp[s]
            print(f'True Positive to False Positive Ratio:{ratio}')
            accuracy = (tp[s]+tn[s])/(tp[s]+tn[s]+fp[s]+fn[s])
            print(f'Accuracy:{accuracy*100}%')
            if ratio > max_ratio and accuracy > max_accuracy:
                max_ratio = ratio
                max_accuracy = accuracy
                ot = s
    return ot

#Get Optimal Theta
optimalTheta = getOptimalTheta(rocSlope,truePositives,falsePositives,trueNegatives,falseNegatives)

#Reshape Initial & Final Weights
final_weights_reshaped = final_weight.reshape(28, 28)
initial_weights_reshaped = initialWeights.reshape(28, 28)

#Function to plot Initial & Final Weight
def showWeightComparions(iwt,fwt):
    fig, axes = plt.subplots(ncols=2, figsize=(8, 4),num='Postsynaptic-gated Weight Comparison')
    ax1, ax2 = axes
    
    ax1.set_title('Figure 3a Postsynaptic-gated Initial Weights')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    ax2.set_title('Figure 3b Postsynaptic-gated Final Weights')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')                        
                             
    im1 = ax1.matshow(iwt.T)
    im2 = ax2.matshow(fwt.T)

    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)                        
    plt.tight_layout()
    fig.show()

#Display heatmaps
showWeightComparions(initial_weights_reshaped,final_weights_reshaped)

#Function to run challenege simulation
def challengeSimulation(X_challenge,Y_challenge,final_weight,optimal_theta):
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
    print(f'*** Running challenge simulation with optimal theta as {optimal_theta} ***')
    #Initialize to zero
    for i in range(m):#Image loop
        st = 0
        for j in range(n):#Pixel loop
            prod = final_weight[j]*X_challenge[i][j]
            st = st + prod
            #print(f'st:{st} and theta:{theta}')
        if (st > optimal_theta):
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
(q_17,q_07,q_19,q_09) = challengeSimulation(X_challenge_mat.copy(),Y_challenge_mat.copy(),final_weight,optimalTheta)
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
    
    plt.figure(num='Postynaptic-gated Challenge Matrix')
    sns.heatmap(q, annot=True,cbar=False, cmap='coolwarm')
    plt.title('Figure 4 Postsynaptic-gated Challenge Matrix')
    plt.xticks(ticks=[0.5,1.5],labels=['Q0','Q1'])
    plt.yticks(ticks=[0.5,1.5],labels=['Q7','Q9'])
    plt.show()

#Call Display Challenge Matrix
displayChallengeMatrix(q)


# ### *****Begin presynaptically-gated scenario***** ###

#Function to run train simulation for presynaptically-gated scenario
def preTrainSimulation(X_train,Y_train,epoch,eta,initialWeights):
    #Gather stats
    #stats = {}
    weights = []
    #img_list = []
    print(f'Epochs:{epoch} and Eta:{eta}')
    #print(f'Learning Input Shape:{Y_train.shape}')
    #Number of features aka pixels
    n = len(X_train[0])
    print(f'Total Features:{n}')
    #Initialize weights
    w = initialWeights
    #Total Train Images
    m = len(X_train)
    #print(f'Training Set Shape:{X_train.shape}')
    print(f'Total Images:{m}')
    for i in range(epoch):
        print(f'*** Running presynaptically-gated train simulation for epoch {i+1} ***')
        for img in range(m):
            #print(f'Processing Image {img+1}')
            st = 0
            zt = Y_train[img]
            yt = zt
            prod = 0
            for j in range(n):
                prod = w[j]*X_train[img][j]
                st = st + prod
            #Update weight
            w = w + eta*X_train[img]*(yt-w)
            #Gather stats
            #img_list.append((st,yt))
        weights.append(w)    
        #stats[i]=img_list
    #return (stats,weights)
    return weights

#Run presynaptically-gated train simulation
simul_weights_pre = []
simul_weights_pre = preTrainSimulation(X_train_mat.copy(),Y_train_mat.copy(),epochs,eta,initialWeights)

#Set last used weight as final weight
final_weight_pre = simul_weights_pre[-2]
truePositives_pre = []
trueNegatives_pre = []
falsePositives_pre = []
falseNegatives_pre = []
#Gets metrics
(truePositives_pre,trueNegatives_pre,falsePositives_pre,falseNegatives_pre) = testSimulation(X_test_mat.copy(),Y_test_mat.copy(),final_weight_pre,theta_max)

#Get performance metrics
pr_pre = []
re_pre = []
fs_pre = []
(pr_pre,re_pre,fs_pre) = getPerformanceMetrics(truePositives_pre,trueNegatives_pre,falsePositives_pre,falseNegatives_pre)

#Function to plot precision, recall and F1 score for presynaptically-gated scenario
def plotMetrics_pre(precision,recall,f1,precision_pre,recall_pre,f1_pre):
    
    T = [i for i in range(len(precision))]
    fig,axes = plt.subplots(ncols=2, figsize=(10,5), num='Precision, Recall and F1 Score Comparison')
    
    ax1, ax2 = axes
    
    ax1.set_title('Figure 5a Postsynaptic-gated Performance')
    ax1.set_xlabel('Theta')
    ax1.set_ylabel('Metrics')

    ax1.plot(T,precision,label = 'Precision')
    ax1.plot(T,recall,label = 'Recall')
    ax1.plot(T,f1,label = 'F1 Score')
    
    ax1.legend()
    
    ax2.set_title('Figure 5b Presynaptic-gated Performance')
    ax2.set_xlabel('Theta')
    ax2.set_ylabel('Metrics')

    ax2.plot(T,precision_pre,label = 'Precision')
    ax2.plot(T,recall_pre,label = 'Recall')
    ax2.plot(T,f1_pre,label = 'F1 Score')
    
    ax2.legend()
    
    plt.tight_layout()
    fig.show()

#Plot Performance Metrics comparison
plotMetrics_pre(pr,re,fs,pr_pre,re_pre,fs_pre)

#Function to plot ROC for presynaptically-gated scenario
def plotROC_pre(tp,fp,tp_pre,fp_pre):
    
    fig,axes = plt.subplots(ncols=2, figsize=(10,5), num='ROC Curve Comparison')
    
    ax1, ax2 = axes
    
    ax1.set_title('Figure 6a Postynaptic-gated ROC Curve')
    ax1.set_xlabel('False Positives')
    ax1.set_ylabel('True Positives')

    ax1.plot(fp,tp,marker='o',markerfacecolor='red')
    
    ax2.set_title('Figure 6b Presynaptic-gated ROC Curve')
    ax2.set_xlabel('False Positives')
    ax2.set_ylabel('True Positives')

    ax2.plot(fp_pre,tp_pre,marker='o',markerfacecolor='red')
    
    plt.tight_layout()
    
    fig.show()

#Plot Performance Comparison
plotROC_pre(truePositives,falsePositives,truePositives_pre,falsePositives_pre)

#Get slope
rocSlope_pre = findROCSlope(truePositives_pre,falsePositives_pre)

#Get Optimal Theta
optimalTheta_pre = getOptimalTheta(rocSlope_pre,truePositives_pre,falsePositives_pre,trueNegatives_pre,falseNegatives_pre)

#Reshape Initial & Final Weights
final_weights_reshaped_pre = final_weight_pre.reshape(28, 28)
initial_weights_reshaped = initialWeights.reshape(28, 28)

#Display initial and final weights for both pre & post synaptically-gated scenarios
def showWeightComparions_pre(iwt,fwt,fwt_pre):
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5),num='Weight Comparison')
    ax1, ax2, ax3 = axes
    
    ax1.set_title('Figure 7a Initial Weights')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    ax2.set_title('Figure 7b Postsynaptic-gated Weights')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y') 
    
    ax3.set_title('Figure 7c Presynaptic-gated Weights')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')    
                             
    im1 = ax1.matshow(iwt.T)
    im2 = ax2.matshow(fwt.T)
    im3 = ax3.matshow(fwt_pre.T)
    
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    fig.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    fig.show()

#Display heatmaps
showWeightComparions_pre(initial_weights_reshaped,final_weights_reshaped,final_weights_reshaped_pre)

q_19_pre = 0
q_09_pre = 0
q_17_pre = 0
q_07_pre = 0
#Get challenge stats
(q_17_pre,q_07_pre,q_19_pre,q_09_pre) = challengeSimulation(X_challenge_mat.copy(),Y_challenge_mat.copy(),final_weight_pre,optimalTheta_pre)
# Creates 2x2
challenege_matrix_pre = [[0 for x in range(w)] for y in range(h)] 
challenege_matrix_pre[0][0] = q_07_pre
challenege_matrix_pre[0][1] = q_17_pre
challenege_matrix_pre[1][0] = q_09_pre
challenege_matrix_pre[1][1] = q_19_pre

q_pre = pd.DataFrame(challenege_matrix_pre)

#Display Challenge Matrix 
def displayChallengeMatrix_pre(q_pre):

    plt.figure(num='Presynaptic-gated Challenge Matrix')
    sns.heatmap(q_pre, annot=True,cbar=False, cmap='coolwarm')
    plt.title('Figure 8 Presynaptic-gated Challenge Matrix')
    plt.xticks(ticks=[0.5,1.5],labels=['Q0','Q1'])
    plt.yticks(ticks=[0.5,1.5],labels=['Q7','Q9'])
    
    plt.show()

#Display Challenge Matrix Comparions
displayChallengeMatrix_pre(q_pre)