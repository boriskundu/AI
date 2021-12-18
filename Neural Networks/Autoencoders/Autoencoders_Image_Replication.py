# ## Autoencoders for image replication.
# ## ***Author - Boris Kundu*** ##

#Import packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import time
import random
from sklearn.utils import shuffle

#Read image features from MNISTnumImages5000_balanced
X = pd.read_csv('MNISTnumImages5000_balanced.txt',delimiter='\t',header=None)
#Print image feature shape
print(f'\nImage feature shape: {X.shape}')
#Show image feature head
X.head()

#Display image feature Info
X.info()

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
#Check head after adding a bias feature and target variable label.
XM.head()

#Number of features inputs
n = len(XM.columns) - 1

#Function returns randomly generated training set of size 4000 and testing set of size 1000 data points.
#It also ensures euqal data points of all 10 classes (0 to 9 digits) for both Training & Testing sets.
def getRandomTrainTestSet(full_df,frac):
    #Get data points for all classes
    X0 = full_df[full_df['label']==0]
    X1 = full_df[full_df['label']==1]
    X2 = full_df[full_df['label']==2]
    X3 = full_df[full_df['label']==3]
    X4 = full_df[full_df['label']==4]
    X5 = full_df[full_df['label']==5]
    X6 = full_df[full_df['label']==6]
    X7 = full_df[full_df['label']==7]
    X8 = full_df[full_df['label']==8]
    X9 = full_df[full_df['label']==9]
    
    #Get train (80%) and test (20%) data points for each class
    X0_Train = X0.sample(frac=frac,axis=0)
    X0_Test = pd.concat([X0, X0_Train]).loc[X0.index.symmetric_difference(X0_Train.index)]
    
    X1_Train = X1.sample(frac=frac,axis=0)
    X1_Test = pd.concat([X1, X1_Train]).loc[X1.index.symmetric_difference(X1_Train.index)]
    
    X2_Train = X2.sample(frac=frac,axis=0)
    X2_Test = pd.concat([X2, X2_Train]).loc[X2.index.symmetric_difference(X2_Train.index)]
    
    X3_Train = X3.sample(frac=frac,axis=0)
    X3_Test = pd.concat([X3, X3_Train]).loc[X3.index.symmetric_difference(X3_Train.index)]
    
    X4_Train = X4.sample(frac=frac,axis=0)
    X4_Test = pd.concat([X4, X4_Train]).loc[X4.index.symmetric_difference(X4_Train.index)]
    
    X5_Train = X5.sample(frac=frac,axis=0)
    X5_Test = pd.concat([X5, X5_Train]).loc[X5.index.symmetric_difference(X5_Train.index)]
    
    X6_Train = X6.sample(frac=frac,axis=0)
    X6_Test = pd.concat([X6, X6_Train]).loc[X6.index.symmetric_difference(X6_Train.index)]
    
    X7_Train = X7.sample(frac=frac,axis=0)
    X7_Test = pd.concat([X7, X7_Train]).loc[X7.index.symmetric_difference(X7_Train.index)]
    
    X8_Train = X8.sample(frac=frac,axis=0)
    X8_Test = pd.concat([X8, X8_Train]).loc[X8.index.symmetric_difference(X8_Train.index)]
    
    X9_Train = X9.sample(frac=frac,axis=0)
    X9_Test = pd.concat([X9, X9_Train]).loc[X9.index.symmetric_difference(X9_Train.index)]
    
    #Create complete balanced Train and Test sets.
    X_Train_full = pd.concat([X0_Train,X1_Train,X2_Train,X3_Train,X4_Train,X5_Train,X6_Train,X7_Train,X8_Train,X9_Train])
    X_Test_full = pd.concat([X0_Test,X1_Test,X2_Test,X3_Test,X4_Test,X5_Test,X6_Test,X7_Test,X8_Test,X9_Test])
    
    X_train = X_Train_full.drop(['label'],inplace=False,axis=1).reset_index(drop=True)
    Y_train = X_Train_full['label'].reset_index(drop=True)
    X_test = X_Test_full.drop(['label'],inplace=False,axis=1).reset_index(drop=True)
    Y_test = X_Test_full['label'].reset_index(drop=True)
    
    return(X_train.copy(),Y_train.copy(),X_test.copy(),Y_test.copy())

#Calculate Sigmoid
def getSigmoid(val):
    sigmoid = (1/(1+np.exp(-val)))
    return sigmoid
#Calculate Sigmoid Derivative
def getSignmodDerivative(val):
    sigmoid = getSigmoid(val)
    sigmoidDerivative = (1-sigmoid)*sigmoid
    return sigmoidDerivative
#Initialize weights
def initializeWeights(feature_count):
    limit = np.sqrt(3/feature_count)
    initialWeights = np.random.uniform(-limit, limit,feature_count)
    return initialWeights
def getJ2loss (expected,predicted):
    return (np.sum(np.square(np.subtract(expected,predicted))))/2.0     

warnings.filterwarnings('ignore')

#Below class represents a Neuron
class Neuron:
    #Initialize 
    def __init__(self,weights,eta=0.01,alpha=0.01,activFunc=getSigmoid,derivFunc=getSignmodDerivative):
        self.weights = weights #Weights of features
        self.weightChange = np.zeros(len(weights))
        self.eta = eta #learning rate
        self.alpha = alpha #Momentum rate
        self.activation = activFunc #Activation function
        self.activationDerivative = derivFunc #Activation derivative function
    #Setters
    def setWeights(self,weights):
        self.weights = weights
    def setOriginalWeights(self,orgWeights):
        self.originalWeights = orgWeights
    def setEta(self,eta):
        self.eta = eta
    def setAlpha(self,alpha):
        self.alpha = alpha
    def setCurrentInput(self,currentInput):
        self.currentInput = currentInput
        self.originalWeights = self.weights
        o = self.getActivationFunction()
        d = self.getActivationDerivative()
    def setCurrentOutput(self, currentOutput):
        self.currentOutput = currentOutput
    def setCurrentOutputDerivative(self, currentOutDeriv):
        self.currentOutputDerivative = currentOutDeriv
    def setActivationFunction(self,activFunc):
        self.activation = activFunc
    def setActivationDerivative(self,derivFunc):
        self.activationDerivative = derivFunc
    def setDelta(self,delta):
        self.delta = delta
    def setWeightedSum(self,weightedSum):
        self.weightedSum = weightedSum
    def setError(self,error):
        self.error = error
    def setWeightChange(self,weightChange):
        self.weightChange = weightChange
    #Getters
    def getWeightChange(self):
        return self.weightChange
    def getOriginalWeight(self):
        return self.originalWeights
    def getWeights(self):
        return self.weights
    def getEta(self):
        return self.eta
    def getAlpha(self):
        return self.alpha
    def getWeightedSum(self):
        weightedSum = np.dot(self.weights,self.currentInput)
        self.setWeightedSum(weightedSum)
        return self.weightedSum
    def getCurrentInput(self):
        return self.currentInput
    def getCurrentOutput(self):
        return self.currentOutput
    def getCurrentOutputDerivative(self):
        return self.currentOutputDerivative
    def getActivationFunction(self):#Call activation function
        self.setCurrentOutput(self.activation(self.getWeightedSum()))
        return self.getCurrentOutput()
    def getActivationDerivative(self):
        self.currentOutputDerivative = self.activationDerivative(self.getWeightedSum())
        return self.getCurrentOutputDerivative()
    def getDelta(self):
        return self.delta
    def getError(self,expected):
        self.setError(expected - self.currentOutput)
        return self.error
    def getHiddenDelta(self,weightedDeltaSum):
        self.setDelta(self.currentOutputDerivative*weightedDeltaSum)
        return self.delta
    def getOutputDelta(self):
        self.setDelta(self.currentOutputDerivative*self.error)
        return self.delta
    def updateWeights(self):
        #next_weight_change = 0
        next_weight_change = self.eta*self.delta*self.currentInput
        momentum = self.alpha*self.weightChange
        self.setWeights(self.weights + next_weight_change + momentum)
        self.setWeightChange(next_weight_change)
    def getWeightedDelta(self):
        return self.weightedDelta
    def calculateWeightedDelta(self):
        weightedDelta = self.originalWeights * self.delta
        self.setWeightedDelta(weightedDelta)
    def setWeightedDelta(self,weightedDelta):
        self.weightedDelta = weightedDelta


# In[359]:


#Take user input
h = int(input('Enter the number of hidden layers:'))

#Neural network layers
layers = []
#Get neurons for each layer
for l in range(h):
    neurons = int(input(f'Enter the number of neurons for hidden layer {l+1}:'))
    layers.append(neurons)
#Add output layer with 784 neurons
layers.append(784)
total_layers = len(layers)
output_layer_index = total_layers - 1

#Make (inputs,outputs) pairs for each layer
layer_sizes = []
for l in range(total_layers):
    if l == 0:
        layer_sizes.append((n,layers[0]))
    else:
        layer_sizes.append((layers[l-1],layers[l]))

class NeuralNetwork:
    #Initialize network
    def __init__(self,layers,layer_sizes,eta=0.01,alpha=0.01,activFunc=getSigmoid,derivFunc=getSignmodDerivative):
        self.layers = layers #Network layers
        self.layer_sizes = layer_sizes #Current size of layers
        self.eta = eta #learning rate
        self.alpha = alpha #Momentum rate
        self.activation = activFunc #Activation function
        self.activationDerivative = derivFunc #Activation derivative function
        self.total_layers = len(layers) #Total layers
        self.output_layer_index = self.total_layers - 1 #Output layers
        self.createNetwork()
    #Create network
    def createNetwork(self):
        neural_network = [] #Neurons in entire network
        for i,o in self.layer_sizes:
            layer_neurons = [] #Neurons in each layer including output
            for k in range(o):
                newNeuron = Neuron(initializeWeights(i),self.eta,self.alpha,self.activation,self.activationDerivative)
                layer_neurons.append(newNeuron)
            neural_network.append(layer_neurons)
            print(i,o)
        self.setNeuralNetwork(neural_network)
    def getNeuralNetwork(self):
        return self.neuralNetwork
    def setNeuralNetwork(self, nn):
        self.neuralNetwork = nn
    def predict(self,X_train,Y_train,X_test,Y_test):
        
        #Digit wise loss
        j2LossTrainDigits = [0 for i in range(10)]
        j2LossTestDigits = [0 for i in range(10)]
        
        X_train_matp = X_train.to_numpy()
        
        Y_train_matp = Y_train.to_numpy()

        X_test_matp = X_test.to_numpy()
        Y_test_matp = Y_test.to_numpy()
        
        total_train_inputs = len(X_train_matp)
        total_test_inputs = len(X_test_matp)
        
        print(f'Total Trainining Data Points:{total_train_inputs}')
        print(f'Total Testing Data Points:{total_test_inputs}')
        
        j2LossTrain = 0
        j2LossTest = 0
        
        predicted_train_images = []
        predicted_test_images = []
        
        expected_output_index_train = 0 #For getting expected(true) output for Train
        expected_output_index_test = 0 #For getting expected(true) output for Test
        
        #Test Predictions - WINNER TAKE All
        for data_point_test in X_test_matp:#For every test point
            loss = 0
            layer_outputs_test = []
            
            for layer in range(self.total_layers): #For each layer 
                output_data_point_test = []
                if layer != output_layer_index: #Hidden Layer 
                    for neuron in self.neuralNetwork[layer]: #Every neuron in current layer
                        if layer == 0: #First hidden layer
                            neuron.setCurrentInput(data_point_test)
                        else: #Other hidden layers
                            neuron.setCurrentInput(layer_outputs_test[layer-1])
                        output_data_point_test.append(neuron.getCurrentOutput())
                else: #Output Layer 784
                    for neuron in self.neuralNetwork[layer]: #Every neuron in current layer
                        neuron.setCurrentInput(layer_outputs_test[layer-1])
                        output_data_point_test.append(neuron.getCurrentOutput())
                layer_outputs_test.append(np.array(output_data_point_test))

            #Compare index of max value for both
            expected_test = Y_test_matp[expected_output_index_test] #Get expected output of data point
            
            testOutNeuron = layer_outputs_test[-1] #Output layer results
            
            #Save predicted image
            predicted_test_images.append(testOutNeuron)
            
            #Get test loss
            loss = getJ2loss(data_point_test,testOutNeuron)
            
            #Sum individual test digit loss
            j2LossTestDigits[expected_test] = j2LossTestDigits[expected_test] + loss
            
            #Get loss for training set
            j2LossTest = j2LossTest + loss
            
            #Get next dp
            expected_output_index_test = expected_output_index_test + 1
            
        #Train Predictions 
        for data_point in X_train_matp:#For every train point
            loss = 0
            layer_outputs = []
            #Train Predictions - Images
            for layer in range(self.total_layers): #For each layer 
                output_data_point = []
                if layer != output_layer_index: #Hidden Layer 
                    for neuron in self.neuralNetwork[layer]: #Every neuron in current layer
                        if layer == 0: #First hidden layer
                            neuron.setCurrentInput(data_point)
                        else: #Other hidden layers
                            neuron.setCurrentInput(layer_outputs[layer-1])
                        output_data_point.append(neuron.getCurrentOutput())
                else: #Output Layer 784
                    for neuron in self.neuralNetwork[layer]: #Every neuron in current layer
                        neuron.setCurrentInput(layer_outputs[layer-1])
                        output_data_point.append(neuron.getCurrentOutput())
                layer_outputs.append(np.array(output_data_point))

            #Compare index of max value for both
            expected = Y_train_matp[expected_output_index_train] #Get expected output of data point
            
            trainOutNeuron = layer_outputs[-1] #Output layer results
            
            #Save predicted image
            predicted_train_images.append(trainOutNeuron)
            
            #Get current data point loss
            loss = getJ2loss(data_point,trainOutNeuron)
            
            #Sum individual digit loss
            j2LossTrainDigits[expected] = j2LossTrainDigits[expected] + loss
            
            #Add to total training set loss
            j2LossTrain = j2LossTrain + loss
            
            #Get next dp
            expected_output_index_train = expected_output_index_train + 1
        
        return (j2LossTrain/total_train_inputs,j2LossTest/total_test_inputs,(np.array(j2LossTrainDigits))/total_train_inputs,(np.array(j2LossTestDigits))/total_test_inputs,predicted_train_images,predicted_test_images)
        
    def feedForward_backPropogate(self,X_train,epoch):
        
        j2LossTrain = []
        
        predicted_train_images = []
        
        X_train_mat = X_train.to_numpy()
        
        for e in range(epoch):
            j2lossEpoch = 0

            X_train_matt = shuffle(X_train_mat)

            total_train_inputs = len(X_train_matt)
            
            print(f'*** Running EPOCH:{e+1} ***')
            start_time = time.time()
        
            #Train & Test Simulation
            for index in range(len(X_train_matt)): #For every train data point
                
                layer_outputs = []
                
                data_point = X_train_matt[index]
                
                #Feed forward Train
                for layer in range(self.total_layers): #For each layer 50,784
                    output_data_point = []
                    if layer != output_layer_index: #Hidden Layer
                        for neuron in self.neuralNetwork[layer]: #Every neuron in current layer
                            if layer == 0: #First hidden layer
                                neuron.setCurrentInput(data_point)
                            else: #Other hidden layers
                                neuron.setCurrentInput(layer_outputs[layer-1])
                            output_data_point.append(neuron.getCurrentOutput())
                    else: #Output Layer 784
                        
                        o = 0
                        for neuron in self.neuralNetwork[layer]: #Every neuron in output layer
                            output_delta = 0
                            neuron.setCurrentInput(layer_outputs[layer-1])
                            output = neuron.getCurrentOutput()
                            output_data_point.append(output)
                            expected_value = data_point[o] #Expected
                            error = neuron.getError(expected_value)
                            output_delta = neuron.getOutputDelta() #Output neuron delta
                            neuron.updateWeights()
                            neuron.calculateWeightedDelta() #Populate weighted delta of output neurons for back propogation
                            o = o + 1 #Get expected output for next neuron in output layer
                    layer_outputs.append(np.array(output_data_point))
                
                trainOutNeuron = layer_outputs[-1] #Output layer results
    
                #Get train loss
                j2lossEpoch = j2lossEpoch + getJ2loss(data_point,trainOutNeuron)
    
                #Back propogate
                for layer in range(output_layer_index,0,-1): #From top layer 784->50
                    k = 0
                    for lowerNeuron in self.neuralNetwork[layer-1]: #Every neuron in lower layer
                        weightedDeltaSum = 0
                        for upperNeuron in self.neuralNetwork[layer]: #Every neuron in upper layer
                            weightedDeltaUp = upperNeuron.getWeightedDelta()
                            weightedDeltaSum = weightedDeltaSum + weightedDeltaUp[k]
                        hiddenDelta = lowerNeuron.getHiddenDelta(weightedDeltaSum) #Hidden neuron delta
                        lowerNeuron.calculateWeightedDelta() #Populate weighted delta for next lower layer
                        lowerNeuron.updateWeights() #Update weights
                        k = k + 1 #For next neuron in current hidden layer
                
                #if e == epoch-1 and index == len(X_train_matt) - 1:
                    #Display input and output images
                    #displayImages(data_point,trainOutNeuron)
                    
            j2LossTrain.append(j2lossEpoch)
            print(f'Average J2 Training Loss:{j2lossEpoch/total_train_inputs}')
            print("--- Execution: %s seconds ---" % (time.time() - start_time))
            
            #Break if J2 training loss fraction for current epoch reaches below 0.001
            if (j2lossEpoch/total_train_inputs <= 1.0):
                print('Stop Training - Average J2 Loss is below 1.0 for current epoch')
                break
            
        return (np.array(j2LossTrain))/total_train_inputs

#Function to plot J2 error for Train
def plotJ2ErrorLoss(j2loss):
    
    E = [i+1 for i in range(len(j2loss))]
    fig,axes = plt.subplots(figsize=(10,5), num='J2 Loss')
    
    axes.set_title('Figure 1 Training Loss vs Epoch')
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Average J2 Loss (per data point)')

    axes.plot(E,j2loss)
    fig.savefig('Average_J2_Training_Loss_Epoch.png')
    fig.show()


#Display digit images
def displayImages(inp,out):                  
    #Reshape each row value to get a 28x28 image out of it
    inp = (inp.reshape(28,28)).T
    out = (out.reshape(28,28)).T
    #Display few images
    plt.figure()
    plt.title('Input Image')
    sns.heatmap(inp)
    plt.figure()
    plt.title('Output Image')
    sns.heatmap(out)
    plt.show()

#Create NeuralNetwork
backpropnn = NeuralNetwork(layers,layer_sizes,0.01,0.01,getSigmoid,getSignmodDerivative)

#Create Training and testing sets
(X_train,Y_train,X_test,Y_test) = getRandomTrainTestSet(XM,0.8)

#Prediction before training
(j2LossTrain_i,j2LossTest_i,j2LossTrainDigits_i,j2LossTestDigits_i,predicted_train_images_i,predicted_test_images_i)  = backpropnn.predict(X_train,Y_train,X_test,Y_test)

#Train Network
epoch = 200
j2LossTrain = backpropnn.feedForward_backPropogate(X_train,epoch)

#Plot J2 training error
plotJ2ErrorLoss(j2LossTrain)

#After training
(j2LossTrain_f,j2LossTest_f,j2LossTrainDigits_f,j2LossTestDigits_f,predicted_train_images_f,predicted_test_images_f)  = backpropnn.predict(X_train,Y_train,X_test,Y_test)

#Function to display loss before and after training
def displayLoss(j2LossTrain_i,j2LossTest_i,j2LossTrain_f,j2LossTest_f):
    plt.figure(num='J2 Loss Error Comparison')
    #Prepare data frame
    before = [j2LossTrain_i,j2LossTest_i]
    after = [j2LossTrain_f,j2LossTest_f]
    index = ['Training','Testing']
    X_axis = np.arange(len(index))
    plt.bar(X_axis - 0.2, before, 0.4, label = 'Before Training')
    plt.bar(X_axis + 0.2, after, 0.4, label = 'After Training')
    plt.xticks(X_axis, index)
    plt.title('Figure 2 J2 Loss Error Comparison')
    plt.xlabel("Dataset")
    plt.ylabel('Average J2 Loss Error (per data point)')
    plt.legend()
    plt.savefig('Average_J2_Train_Test_Loss_Pre_Post_Training.png')
    plt.show()

#Function to display digit wise loss before and after training
def displayDigitLoss(j2LossTrainDigits_i,j2LossTestDigits_i,j2LossTrainDigits_f,j2LossTestDigits_f):
    plt.figure(num = 'J2 Train Digit Loss Error Comparison')
    #Prepare train data frame
    before = [j2LossTrainDigits_i[0],j2LossTrainDigits_i[1],j2LossTrainDigits_i[2],
              j2LossTrainDigits_i[3],j2LossTrainDigits_i[4],j2LossTrainDigits_i[5],
              j2LossTrainDigits_i[6],j2LossTrainDigits_i[7],j2LossTrainDigits_i[8],
              j2LossTrainDigits_i[9]]
    after = [j2LossTrainDigits_f[0],j2LossTrainDigits_f[1],j2LossTrainDigits_f[2],
              j2LossTrainDigits_f[3],j2LossTrainDigits_f[4],j2LossTrainDigits_f[5],
              j2LossTrainDigits_f[6],j2LossTrainDigits_f[7],j2LossTrainDigits_f[8],
              j2LossTrainDigits_f[9]]
    index = ['0','1','2','3','4','5','6','7','8','9']
    X_axis = np.arange(len(index))
    plt.bar(X_axis - 0.2, before, 0.4, label = 'Before Training')
    plt.bar(X_axis + 0.2, after, 0.4, label = 'After Training')
    plt.xticks(X_axis, index)
    plt.title('Figure 3 J2 Train Digit Loss Error Comparison')
    plt.xlabel("Train Digits")
    plt.ylabel('Average J2 Train Digit Loss Error(per data point)')
    plt.legend()
    plt.savefig('Average_J2_Train_Digit_Loss.png')
    plt.show()
    
    plt.figure(num = 'J2 Test Digit Loss Error Comparison')
    #Prepare test data frame
    before = [j2LossTestDigits_i[0],j2LossTestDigits_i[1],j2LossTestDigits_i[2],
              j2LossTestDigits_i[3],j2LossTestDigits_i[4],j2LossTestDigits_i[5],
              j2LossTestDigits_i[6],j2LossTestDigits_i[7],j2LossTestDigits_i[8],
              j2LossTestDigits_i[9]]
    after = [j2LossTestDigits_f[0],j2LossTestDigits_f[1],j2LossTestDigits_f[2],
              j2LossTestDigits_f[3],j2LossTestDigits_f[4],j2LossTestDigits_f[5],
              j2LossTestDigits_f[6],j2LossTestDigits_f[7],j2LossTestDigits_f[8],
              j2LossTestDigits_f[9]]

    plt.bar(X_axis - 0.2, before, 0.4, label = 'Before Training')
    plt.bar(X_axis + 0.2, after, 0.4, label = 'After Training')
    plt.xticks(X_axis, index)
    plt.title('Figure 4 J2 Test Digit Loss Error Comparison')
    plt.xlabel("Test Digits")
    plt.ylabel('Average J2 Test Digit Loss Error (per data point)')
    plt.legend()
    plt.savefig('Average_J2_Test_Digit_Loss.png')
    plt.show()

#Get weights from network
def getWeightsFromNetwork(network):
    hidden_weights = []
    neuralNets = network.getNeuralNetwork()
    for layer in range(total_layers): #For each layer 50,784
        if layer != output_layer_index: #Hidden Layers
            for neuron in neuralNets[layer]: #Every neuron in current hidden layer
                if layer == 0: #First hidden layer
                    hidden_weights.append(neuron.getWeights())
    return hidden_weights

#Plot loss
displayLoss(j2LossTrain_i,j2LossTest_i,j2LossTrain_f,j2LossTest_f)

#Plot digit loss
displayDigitLoss(j2LossTrainDigits_i,j2LossTestDigits_i,j2LossTrainDigits_f,j2LossTestDigits_f)

#Display hidden images
def displayHiddenImages(auto,clas,num):                  
    new = []
    old = []

    #Reshape each row value to get a 28x28 image out of it
    for i in range(num):
        new.append(auto[i].reshape(28,28))
        old.append(clas[i].reshape(28,28))
    
    fig1, axes1 = plt.subplots(nrows=4, ncols=5, figsize=(20, 10),num='Autoencoder Hidden Neruron Features')

    axes1[0][0].imshow(new[0],cmap='gist_heat')
    axes1[0][1].imshow(new[1],cmap='gist_heat')
    axes1[0][2].imshow(new[2],cmap='gist_heat')
    axes1[0][3].imshow(new[3],cmap='gist_heat')
    axes1[0][4].imshow(new[4],cmap='gist_heat')
    
    axes1[1][0].imshow(new[5],cmap='gist_heat')
    axes1[1][1].imshow(new[6],cmap='gist_heat')
    axes1[1][2].imshow(new[7],cmap='gist_heat')
    axes1[1][3].imshow(new[8],cmap='gist_heat')
    axes1[1][4].imshow(new[9],cmap='gist_heat')
    
    axes1[2][0].imshow(new[10],cmap='gist_heat')
    axes1[2][1].imshow(new[11],cmap='gist_heat')
    axes1[2][2].imshow(new[12],cmap='gist_heat')
    axes1[2][3].imshow(new[13],cmap='gist_heat')
    axes1[2][4].imshow(new[14],cmap='gist_heat')
    
    axes1[3][0].imshow(new[15],cmap='gist_heat')
    axes1[3][1].imshow(new[16],cmap='gist_heat')
    axes1[3][2].imshow(new[17],cmap='gist_heat')
    axes1[3][3].imshow(new[18],cmap='gist_heat')
    axes1[3][4].imshow(new[19],cmap='gist_heat')

    fig1.suptitle('Figure 5 Autoencoder Hidden Neruron Features')
    fig1.savefig('Autoencoder_Hidden_Neruron_Features.png')
    plt.tight_layout()
    fig1.show()

    fig2, axes2 = plt.subplots(nrows=4, ncols=5, figsize=(20, 10),num='Multiple Classifier Hidden Neruron Features')

    axes2[0][0].imshow(old[0],cmap='gist_heat')
    axes2[0][1].imshow(old[1],cmap='gist_heat')
    axes2[0][2].imshow(old[2],cmap='gist_heat')
    axes2[0][3].imshow(old[3],cmap='gist_heat')
    axes2[0][4].imshow(old[4],cmap='gist_heat')
    
    axes2[1][0].imshow(old[5],cmap='gist_heat')
    axes2[1][1].imshow(old[6],cmap='gist_heat')
    axes2[1][2].imshow(old[7],cmap='gist_heat')
    axes2[1][3].imshow(old[8],cmap='gist_heat')
    axes2[1][4].imshow(old[9],cmap='gist_heat')
    
    axes2[2][0].imshow(old[10],cmap='gist_heat')
    axes2[2][1].imshow(old[11],cmap='gist_heat')
    axes2[2][2].imshow(old[12],cmap='gist_heat')
    axes2[2][3].imshow(old[13],cmap='gist_heat')
    axes2[2][4].imshow(old[14],cmap='gist_heat')
    
    axes2[3][0].imshow(old[15],cmap='gist_heat')
    axes2[3][1].imshow(old[16],cmap='gist_heat')
    axes2[3][2].imshow(old[17],cmap='gist_heat')
    axes2[3][3].imshow(old[18],cmap='gist_heat')
    axes2[3][4].imshow(old[19],cmap='gist_heat')

    fig2.suptitle('Figure 6 Multi-Classifier Hidden Neruron Features')
    
    fig2.savefig('MultiClassifier_Hidden_Neruron_Features.png')
    plt.tight_layout()
    fig2.show()
    plt.show()

hidden_weights = getWeightsFromNetwork(backpropnn)

#Read hidden neuron weights from multiple classifier
classifierHiddenWeights = pd.read_csv('MultiClassifierWeights.csv',header=None)

#Drop bias feature weights
classifierHiddenWeights = classifierHiddenWeights.iloc[: , :-1]

classifierHiddenWeights_np = classifierHiddenWeights.to_numpy()

#Display hidden neuron features
displayHiddenImages(hidden_weights,classifierHiddenWeights_np,20)

#Function to save final network weights
def saveWeightsToCSV(weights,file_name):
    full_name = file_name+'.csv'
    np.savetxt(full_name, weights, delimiter=",")

#Save final network weights
saveWeightsToCSV(hidden_weights,'AutoencoderWeights')

#Display hidden images
def displayOutputImages(org,pred,num):                  
    ini = []
    out = []

    #Reshape each row value to get a 28x28 image out of it
    for i in range(num):
        ini.append(org[i].reshape(28,28).T)
        out.append(pred[i].reshape(28,28).T)
    
    fig1, axes1 = plt.subplots(ncols=8, figsize=(20,5),num='Autoencoder Test Input')

    axes1[0].imshow(ini[0],cmap='gist_heat')
    axes1[1].imshow(ini[1],cmap='gist_heat')
    axes1[2].imshow(ini[2],cmap='gist_heat')
    axes1[3].imshow(ini[3],cmap='gist_heat')
    axes1[4].imshow(ini[4],cmap='gist_heat')
    axes1[5].imshow(ini[5],cmap='gist_heat')
    axes1[6].imshow(ini[6],cmap='gist_heat')
    axes1[7].imshow(ini[7],cmap='gist_heat')

    fig1.suptitle('Figure 7 Test Input Images')
    
    fig1.savefig('Test_Input_Images.png')
    plt.tight_layout()
    fig1.show()
    
    fig2, axes2 = plt.subplots(ncols=8, figsize=(20,5),num='Autoencoder Test Output')

    axes2[0].imshow(out[0],cmap='gist_heat')
    axes2[1].imshow(out[1],cmap='gist_heat')
    axes2[2].imshow(out[2],cmap='gist_heat')
    axes2[3].imshow(out[3],cmap='gist_heat')
    axes2[4].imshow(out[4],cmap='gist_heat')
    axes2[5].imshow(out[5],cmap='gist_heat')
    axes2[6].imshow(out[6],cmap='gist_heat')
    axes2[7].imshow(out[7],cmap='gist_heat')

    fig2.suptitle('Figure 8 Test Output Images')

    fig2.savefig('Test_Output_Images.png')
    plt.tight_layout()
    fig2.show()
    plt.show()

#Convert predicted images to data frame
predicted_test_final = pd.DataFrame(predicted_test_images_f)
choice = (np.array(random.sample(range(10), 8))*100).tolist()
pred_test = predicted_test_final.iloc[choice].to_numpy()
exp_test = X_test.iloc[choice].to_numpy()

#Display hidden neuron features
displayOutputImages(exp_test,pred_test,8)