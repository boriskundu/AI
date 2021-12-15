# **A Perceptron to classify randomly generated linearly-separable two-class data**
# 
# Authors - Boris Kundu

#Import packages
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

#Function to assign classes
def getFeatureClasses():
    pos = 0
    neg = 0
    while (pos!=100 and neg!=100):
        pos = 0
        neg = 0
        #Generate 2 random feature with 200 data points between -40 and 40
        X1 = np.random.randint(-40,40,200)
        X2 = np.random.randint(-40,40,200)
        Y = []
        Y_calculated = X1 + (3*X2) - 2
        for val in Y_calculated:
            if val > 0:
                Y.append(1)
                pos = pos + 1
            else:
                Y.append(-1)
                neg = neg + 1
    return (X1,X2,Y)

#Get target output class
(X1,X2,Y) = getFeatureClasses()

#Create dataframe
df = pd.DataFrame(X1, columns=['X1'])

#Add other features and target to our master dataframe
df['X2'] = X2
df['Y'] = Y

#Add bias term/feature
bias = np.ones(len(X1))
df['bias'] = bias

#Check head
df.head()

#Define our feature and target dataframes
X = df[['X1','X2','bias']]
Y = df[['Y']]

#Function to run Stochastic Delta train simulation
def batchDeltaTrain(X_train,Y_train,epoch,eta,initialWeights):
    weightUpdateCountBatch = 0
    #Wrong predictions
    trainFrac = 0
    #Save training error for epochs
    trainError = []
    #Weights to return
    weights = []
    print(f'\n\nEpochs:{epoch} and Learning Rate:{eta}')
    #Number of features
    n = len(X_train[0])
    print(f'Total Training Features (including bias):{n}')
    #Initialize weights
    w = initialWeights
    #Total Training data points
    m = len(X_train)
    #Total Testing data points
    print(f'Total Training Points:{m}')
    print(f'Initial Weights:{w}')
    for i in range(epoch):
        #Correctly classified
        correctPredictions = 0
        #Append weights
        weights.append(w)
        trainFrac = 0
        print(f'*** Running batch perceptron train simulation for epoch {i+1} ***')
        st = 0
        loss = 0
        yt = []
        st = np.dot(X_train, w)
        for s in st:
            if s > 0:
                yt.append(1)
            else:
                yt.append(-1)
        yt = np.array(yt).reshape(200,1)
        #Check loss/error
        loss = Y_train - yt
        for loss_indiv in loss:
            if loss_indiv == 0:
                correctPredictions = correctPredictions + 1
            else:
                trainFrac = trainFrac + 1
        dt = eta * np.dot(X_train.T,loss).T[0]
        weightUpdateCountBatch = weightUpdateCountBatch + 1
        #Update weight
        w = w + dt
        trainError.append((trainFrac/m)*100)
        print(f'Train Error Percentage:{trainError[-1]} %')
        #Exit on perfect classficiation for all training points
        #if correctPredictions == m:
        #   print('Stop Training - Perfect Predictions')
        #    return (weights,trainError)
    print('Stop Training - End of Epochs')
    #print(weights)
    return (weights,trainError,weightUpdateCountBatch)

#Numbr of simulations
epoch = 25 
#Learning rate
eta = 0.0001
#Initialize weights
initialWeights = np.random.uniform(0, 0.5, len(X.columns))
#Send matrix form
X_mat = X.to_numpy()
Y_mat = Y.to_numpy()
#Run delta train simulations
(simulWeights,trainErrorFraction,weightUpdateCountBatch25) = batchDeltaTrain(X_mat,Y_mat,epoch,eta,initialWeights)

#Function to display Training Error per Epoch
def plotError(error):
    E = [i+1 for i in range(len(error))]
    fig,axes = plt.subplots(figsize=(10,5),num='Training Error Plot')
    axes.set_title('Figure 1 Training Error vs Epoch')
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Training Error Perfcentage (%)')
    axes.plot(E,error)
    plt.show()

#Display Training Error
plotError(trainErrorFraction)

newEpoch = 100
(newWeights,newTrainErrorFraction,weightUpdateCountBatch100) = batchDeltaTrain(X_mat,Y_mat,newEpoch,eta,initialWeights)

#Function to plot decision boundary
def plotDecisionBoundary(weights,X,Y):
    full = pd.concat([X,Y],axis=1)
    full_neg = full[full['Y']==-1]
    full_pos = full[full['Y']==1]
    #Points for plotting
    x = np.random.randint(-50,50,200)
    
    fig,axes = plt.subplots(figsize=(10,10),num='Decision Boundary')
    
    axes.set_title('Figure 2 Decision Boundary')
    axes.set_xlabel('X1')
    axes.set_ylabel('X2')
    #Plot the two classes
    axes.scatter(full_neg['X1'],full_neg['X2'],label = 'Y=-1')
    axes.scatter(full_pos['X1'],full_pos['X2'],label = 'Y=1')
    max_epochs = len(weights)
    current_epoch = 0
    
    for wt in weights:
        wt = wt.tolist()
        current_epoch = current_epoch + 1
        if current_epoch == 5 or current_epoch == 10 or current_epoch == 50 or current_epoch == 100 or current_epoch == max_epochs:
            #Get slope and y-intercept
            m = -(wt[2] / wt[1]) / (wt[2] / wt[0])
            c = -wt[2] / wt[1]
            #Form equation of a straight line
            y = m*x + c
            if current_epoch!= max_epochs:
                axes.plot(x,y,label = 'Epoch '+str(current_epoch))
            else:
                axes.plot(x,y,label = 'Epoch 100')
    axes.legend()
    plt.show()

#Display decision boundary
plotDecisionBoundary(newWeights,X,Y)

#Function to display Training Error per Learning Rate
def plotErrorLearningRate(error_1,error_2,error_3,error_4):
    E = [i+1 for i in range(len(error_1))]
    fig,axes = plt.subplots(figsize=(10,5),num='Training Error Plot')
    
    axes.set_title('Figure 3 Training Error for Different Learning Rates')
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Training Error Perfcentage (%)')

    axes.plot(E,error_1,label='Learning rate = 0.1')
    axes.plot(E,error_2,label='Learning rate = 0.01')
    axes.plot(E,error_3,label='Learning rate = 0.001')
    axes.plot(E,error_4,label='Learning rate = 0.0001')
    
    axes.legend()
    plt.show()

fiftyEpoch = 50
eta1 = 0.1
eta2 = 0.01
eta3 = 0.001
eta4 = 0.0001
(eta1Weights,eta1Error,weightUpdateCountBatch1) = batchDeltaTrain(X_mat,Y_mat,fiftyEpoch,eta1,initialWeights)
(eta2Weights,eta2Error,weightUpdateCountBatch2) = batchDeltaTrain(X_mat,Y_mat,fiftyEpoch,eta2,initialWeights)
(eta3Weights,eta3Error,weightUpdateCountBatch3) = batchDeltaTrain(X_mat,Y_mat,fiftyEpoch,eta3,initialWeights)
(eta4Weights,eta4Error,weightUpdateCountBatch4) = batchDeltaTrain(X_mat,Y_mat,fiftyEpoch,eta4,initialWeights)

#Display Training Error for different Learning Rates
plotErrorLearningRate(eta1Error,eta2Error,eta3Error,eta4Error)

#Get minimum error
min1 = min(eta1Error)
min2 = min(eta2Error)
min3 = min(eta3Error)
min4 = min(eta4Error)
#Epoch of minimum
pos1 = eta1Error.index(min1) + 1
pos2 = eta2Error.index(min2) + 1
pos3 = eta3Error.index(min3) + 1
pos4 = eta4Error.index(min4) + 1
print(f'\nLearning Rate:{eta1} => Minimum Training Error:{min1}% => Epoch:{pos1}')
print(f'Learning Rate:{eta2} => Minimum Training Error:{min2}% => Epoch:{pos2}')
print(f'Learning Rate:{eta3} => Minimum Training Error:{min3}% => Epoch:{pos3}')
print(f'Learning Rate:{eta4} => Minimum Training Error:{min4}% => Epoch:{pos4}')

df = pd.DataFrame([[eta1, min1,pos1],[eta2, min2,pos2],[eta3, min3,pos3],[eta4, min4,pos4]], columns=['Eta','MTE','Epoch'])

min_MTE_df= df[df['MTE'] == df['MTE'].min()]
best_eta = min_MTE_df[min_MTE_df['Epoch'] == min_MTE_df['Epoch'].min()]
bestEta = best_eta.iloc[0]['Eta']
print(f'Best Learning Rate:{bestEta}')

#Function to run Stochastic Delta train simulation
def stochasticDeltaTrain(X_train,Y_train,epoch,eta,initialWeights):
    #Weight Updates
    weightUpdateCountStoc = 0
    #Wrong predictions
    trainFrac = 0
    #Save training error for epochs
    trainError = []
    #Save testing error for epochs
    #Weights to return
    weights = []
    print(f'\n\nEpochs:{epoch} and Learning Rate:{eta}')
    #Number of features
    n = len(X_train[0])
    print(f'Total Training Features (including bias):{n}')
    #Initialize weights
    w = initialWeights
    #Total Training data points
    m = len(X_train)
    #Total Testing data points
    print(f'Total Training Points:{m}')
    for i in range(epoch):
        #Correctly classified
        correctPredictions = 0
        #Append weights
        weights.append(w)
        trainFrac = 0
        #Predict on Test Set using current weights
        print(f'*** Running stochastic perceptron train simulation for epoch {i+1} ***')
        for pt in range(m):
            st = 0
            loss = 0
            yt = 0
            for j in range(n):
                st = st + w[j]*X_train[pt][j]
            if st > 0:
                yt = 1
            else:
                yt = -1
            #Check loss/error
            loss = Y_train[pt] - yt
            if loss == 0:
                correctPredictions = correctPredictions + 1
            else:
                trainFrac = trainFrac + 1
            weightUpdateCountStoc = weightUpdateCountStoc + 1
            #Update weight
            w = w + eta*loss*X_train[pt]
        trainError.append((trainFrac/m)*100)
        #print(f'Correct Train Predictions:{correctPredictions}')
        print(f'Train Error Percentage:{trainError[-1]} %')
        #Exit on perfect classficiation for all training points
        #if correctPredictions == m:
         #   print('Stop Training - Perfect Predictions')
          #  return (weights,trainError)
    print('Stop Training - End of Epochs')
    return (weights,trainError,weightUpdateCountStoc)

#Get time for stochastic
fiftyEpoch = 50
start_time_stoch = time.time()
(bestEtaWeights,bestEtaError,weightUpdateCountStoc) = stochasticDeltaTrain(X_mat,Y_mat,fiftyEpoch,bestEta,initialWeights)
stoch_exec_secs = (time.time() - start_time_stoch)

#Get time for batch
start_time_batch = time.time()
(bestEtaWeights,bestEtaError,weightUpdateCountBatch) = batchDeltaTrain(X_mat,Y_mat,fiftyEpoch,bestEta,initialWeights)
batch_exec_secs = (time.time() - start_time_batch)

#Display results
print(f'\nEpochs: {fiftyEpoch}')
print(f'Learning Rate: {bestEta}')

print(f'\nStochastic Training:')
print(f'\tExecution Time(secs): {stoch_exec_secs}')
print(f'\tNumber of Weight Updates: {weightUpdateCountStoc}')

print(f'\nBatch Training:')
print(f'\tExecution Time(secs): {batch_exec_secs}')
print(f'\tNumber of Weight Updates: {weightUpdateCountBatch}')

# ## Implementing Delta Train Rule with a decaying learning rate.
# Decaying learning rate can be any value: [0,1].
# For this implementation, we have taken the decay of the learning rate to be "0.8".

#Function to run Stochastic Delta train simulation
def batchDeltaTrain_decayLearning(X_train,Y_train,epoch,eta,initialWeights, decay=1):
    weightUpdateCountBatch = 0
    #Wrong predictions
    trainFrac = 0
    #Storing original learning rate
    original_eta = eta
    #Save training error for epochs
    trainError = []
    #Weights to return
    weights = []
    print(f'\n\nEpochs:{epoch} and Learning Rate:{eta}')
    #Number of features
    n = len(X_train[0])
    print(f'Total Training Features (including bias):{n}')
    #Initialize weights
    w = initialWeights
    #Total Training data points
    m = len(X_train)
    #Total Testing data points
    print(f'Total Training Points:{m}')
    print(f'Initial Weights:{w}')
    for i in range(epoch):
        #Correctly classified
        correctPredictions = 0
        #Append weights
        weights.append(w)
        trainFrac = 0
        print(f'*** Running batch perceptron train simulation with decaying learning rate for epoch {i+1}. ***')
        print(f'Learning Rate: {eta}, Decay in Learning Rate: {decay}')
        st = 0
        loss = 0
        yt = []
        st = np.dot(X_train, w)
        for s in st:
            if s > 0:
                yt.append(1)
            else:
                yt.append(-1)
        yt = np.array(yt).reshape(200,1)
        #Check loss/error
        loss = Y_train - yt
        for loss_indiv in loss:
            if loss_indiv == 0:
                correctPredictions = correctPredictions + 1
            else:
                trainFrac = trainFrac + 1
        dt = eta * np.dot(X_train.T,loss).T[0]
        weightUpdateCountBatch = weightUpdateCountBatch + 1
        #Update weight
        w = w + dt
        #decaying learning rate
        #eta = pow(decay,i) * original_eta
        eta = eta * decay
        trainError.append((trainFrac/m)*100)
        print(f'Train Error Percentage:{trainError[-1]} %')
        #Exit on perfect classficiation for all training points
        #if correctPredictions == m:
        #   print('Stop Training - Perfect Predictions')
        #    return (weights,trainError)
    print('Stop Training - End of Epochs')
    return (weights,trainError,weightUpdateCountBatch)

# ### Trains a model with decaying learning rate.
# You can modify the decay in learning rate from the following block. To change, modify the value of the variable: decay in the block.

#Number of simulations
epoch = 25 
#Learning rate
decayeta = 0.8
#decay in learning rate
decay = 0.8
#Send matrix form
X_mat = X.to_numpy()
Y_mat = Y.to_numpy()
#Run batch decay delta train simulations with decaying learning rate
(simulWeightsDecay,trainErrorDecay,weightUpdateCountBatchDecay25) = batchDeltaTrain_decayLearning(X_mat,Y_mat,epoch,decayeta,initialWeights,decay)
#Run batch delta train simulations with constant learning rate
(simulWeightsNoDecay,trainErrorNoDecay,weightUpdateCountBatchNoDecay25) = batchDeltaTrain(X_mat,Y_mat,epoch,decayeta,initialWeights)

#Function to display Training Error per Epoch
def plotError(eta,errors, labels,title):
    E = [i+1 for i in range(len(errors[0]))]
    fig,axes = plt.subplots(figsize=(10,5),num='Training Error Plot')
    axes.set_title(f'{title} Training Error vs Epoch at initial learning rate: {eta}')
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Training Error Perfcentage (%)')
    for error,label in zip(errors,labels):
        axes.plot(E,error,label=label)
    plt.legend(loc='upper right')
    plt.show()

#Plots comparison between training error at constant rate V/s training error at decaying learning rate
plotError(decayeta,(trainErrorNoDecay, trainErrorDecay),('Training Error at Constant Learing rate','Training Error at Decaying Learing rate'),'Figure 4')

# ## Implementing Delta Train Rule with a adaptive learning rate.
# Threshold [t](to compare the new error with the previous error) for modifying the learning rate can be any value: [0,1].
# Decay in learning rate [d] for decaying the learning rate can be any value: [0,1].
# Improvement in learning rate [D] for improving the learning rate can be any value: [1,infinity).
# 
# For this implementation, we have taken:
# t = 3 (%)
# d = 0.9
# D = 1.02

#Function to run adaptive batch Delta train simulation
def batchDeltaTrain_adaptiveLearning(X_train,Y_train,epoch,eta,initialWeights,t = 0.03, d = 0.9, D = 1.02):
    weightUpdateCountBatch = 0
    #Wrong predictions
    trainFrac = 0
    #Storing original learning rate
    original_eta = eta
    #Save training error for epochs
    trainError = []
    #Weights to return
    weights = []
    print(f'\n\nEpochs:{epoch} and Learning Rate:{eta}')
    #Number of features
    n = len(X_train[0])
    print(f'Total Training Features (including bias):{n}')
    #Initialize weights
    w = initialWeights
    #Total Training data points
    m = len(X_train)
    #Total Testing data points
    print(f'Total Training Points:{m}')
    print(f'Initial Weights:{w}')
    prev_error = 0
    for i in range(epoch):
        #Correctly classified
        correctPredictions = 0
        #Append weights
        weights.append(w)
        trainFrac = 0
        print(f'*** Running batch perceptron train simulation with adaptive learning rate for epoch {i+1}. ***')
        print(f'Learning Rate: {eta}, t : {t}, d: {d}, D: {D}')
        st = 0
        loss = 0
        yt = []
        st = np.dot(X_train, w)
        for s in st:
            if s > 0:
                yt.append(1)
            else:
                yt.append(-1)
        yt = np.array(yt).reshape(200,1)
        #Check loss/error
        loss = Y_train - yt
        for loss_indiv in loss:
            if loss_indiv == 0:
                correctPredictions = correctPredictions + 1
            else:
                trainFrac = trainFrac + 1
        curr_error = (trainFrac/m)*100
        trainError.append(curr_error)
        if i == 0:
            #Update weight
            dt = eta * np.dot(X_train.T,loss).T[0]
            w = w + dt
            weightUpdateCountBatch = weightUpdateCountBatch + 1
        if i != 0:
            if(trainError[-1] - trainError[-2] >= t):
                #decrease learning rate
                eta = d * eta
            else:
                #increase learning rate
                eta = D * eta
                #Update weight
                dt = eta * np.dot(X_train.T,loss).T[0]
                w = w + dt
                weightUpdateCountBatch = weightUpdateCountBatch + 1
        print(f'Train Error Percentage:{trainError[-1]} %')
        #Exit on perfect classficiation for all training points
        #if correctPredictions == m:
        #   print('Stop Training - Perfect Predictions')
        #    return (weights,trainError)
    print('Stop Training - End of Epochs')
    #print(weights)
    return (weights,trainError,weightUpdateCountBatch)

#Number of simulations
epoch = 25 
#Learning rate
adapteta = 0.5
#Initializing hyper-paramters
t = 3 #%
d = 0.9
D = 1.02
#Send matrix form
X_mat = X.to_numpy()
Y_mat = Y.to_numpy()
#Run batch daptive delta train simulations with adaptive learning.
(simulWeightsAdapt,trainErrorAdapt,weightUpdateCountBatchAdapt25) = batchDeltaTrain_adaptiveLearning(X_mat,Y_mat,epoch,adapteta,initialWeights,t,d,D)
#Run batch delta train simulations with constant learning rate
(simulWeightsNoAdapt,trainErrorNoAdapt,weightUpdateCountBatchNoAdapt25) = batchDeltaTrain(X_mat,Y_mat,epoch,adapteta,initialWeights)

plotError(adapteta,(trainErrorNoAdapt, trainErrorAdapt),('Training Error at Constant Learning rate','Training Error at Adaptive Learning rate'),'Figure 5')