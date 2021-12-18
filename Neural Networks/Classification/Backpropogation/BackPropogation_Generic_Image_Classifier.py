# ## Back Propogation
# ## ***Author - Boris Kundu*** ##

#Import packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import time

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
#Add bias feature as 1
XM['bias'] = np.ones(len(XM))
#Check head after adding a bias feature and target variable label.
XM.head()

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

#Dictionary of expected outputs
#Key represents digit in image
#Value represents output layer
y_ideal = {
            0:[1,0,0,0,0,0,0,0,0,0],# Image is 0
            1:[0,1,0,0,0,0,0,0,0,0],# Image is 1
            2:[0,0,1,0,0,0,0,0,0,0],# Image is 2
            3:[0,0,0,1,0,0,0,0,0,0],# Image is 3
            4:[0,0,0,0,1,0,0,0,0,0],# Image is 4
            5:[0,0,0,0,0,1,0,0,0,0],# Image is 5
            6:[0,0,0,0,0,0,1,0,0,0],# Image is 6
            7:[0,0,0,0,0,0,0,1,0,0],# Image is 7
            8:[0,0,0,0,0,0,0,0,1,0],# Image is 8
            9:[0,0,0,0,0,0,0,0,0,1] # Image is 9
          }

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
    def getStandardizedOutput(self,expected,low = 0.25,high = 0.75):
        if self.currentOutput <= low:
            self.currentOutput = 0
        elif self.currentOutput >= high:
            self.currentOutput = 1
        trueError = self.getError(expected)
        return self.currentOutput
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

#Take user input
h = int(input('Enter the number of hidden layers:'))

#Neural network layers
layers = []
#Get neurons for each layer
for l in range(h):
    neurons = int(input(f'Enter the number of neurons for hidden layer {l+1}:'))
    layers.append(neurons)
#Add output layer with 10 neurons
layers.append(10)
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
    def predict(self,full_df):
        (X_trainp,Y_trainp,X_testp,Y_testp) = getRandomTrainTestSet(full_df,0.8)
        
        X_train_matp= X_trainp.to_numpy()
        Y_train_matp = Y_trainp.to_numpy()

        X_test_matp = X_testp.to_numpy()
        Y_test_matp = Y_testp.to_numpy()
        
        total_train_inputs = len(X_train_matp)
        total_test_inputs = len(X_test_matp)
        
        print(f'Total Trainining Data Points:{total_train_inputs}')
        print(f'Total Testing Data Points:{total_test_inputs}')
        
        expected_output_index_train = 0 #For getting expected(true) output for Train
        expected_output_index_test = 0 #For getting expected(true) output for Test
        
        w, h = 10, 10
        confusion_test = [[0 for x in range(w)] for y in range(h)]
        confusion_train = [[0 for x in range(w)] for y in range(h)] 
        
        #Test Predictions - WINNER TAKE All
        for data_point_test in X_test_matp:#For every test point
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
                else: #Output Layer 10
                    for neuron in self.neuralNetwork[layer]: #Every neuron in current layer
                        neuron.setCurrentInput(layer_outputs_test[layer-1])
                        output_data_point_test.append(neuron.getCurrentOutput())
                layer_outputs_test.append(np.array(output_data_point_test))

            #Compare index of max value for both
            expected_test = Y_test_matp[expected_output_index_test] #Get expected output of data point
            expected_output_layer_test = y_ideal.get(expected_test) #Get expected output for all 10 neurons that make up the digit
            expected_output_layer_test = np.array(expected_output_layer_test)
            expected_test_out_max_index = np.where(expected_output_layer_test == expected_output_layer_test.max())

            testOutNeuron = layer_outputs_test[-1] #Output layer results
            test_out_max_index = np.where(testOutNeuron == testOutNeuron.max())
            
            #Increment count in confusion matrix
            confusion_test[expected_test_out_max_index[0][0]][test_out_max_index[0][0]] = confusion_test[expected_test_out_max_index[0][0]][test_out_max_index[0][0]] + 1
            #Get next dp
            expected_output_index_test = expected_output_index_test + 1
            
        #Train Predictions - WINNER TAKE All
        for data_point in X_train_matp:#For every train point
            layer_outputs = []
            #Train Predictions - WINNER TAKE All
            for layer in range(self.total_layers): #For each layer 
                output_data_point = []
                if layer != output_layer_index: #Hidden Layer 
                    for neuron in self.neuralNetwork[layer]: #Every neuron in current layer
                        if layer == 0: #First hidden layer
                            neuron.setCurrentInput(data_point)
                        else: #Other hidden layers
                            neuron.setCurrentInput(layer_outputs[layer-1])
                        output_data_point.append(neuron.getCurrentOutput())
                else: #Output Layer 10
                    for neuron in self.neuralNetwork[layer]: #Every neuron in current layer
                        neuron.setCurrentInput(layer_outputs[layer-1])
                        output_data_point.append(neuron.getCurrentOutput())
                layer_outputs.append(np.array(output_data_point))

            #Compare index of max value for both
            expected = Y_train_matp[expected_output_index_train] #Get expected output of data point
            expected_output_layer = y_ideal.get(expected) #Get expected output for all 10 neurons that make up the digit
            expected_output_layer = np.array(expected_output_layer)
            expected_train_out_max_index = np.where(expected_output_layer == expected_output_layer.max())

            trainOutNeuron = layer_outputs[-1] #Output layer results
            train_out_max_index = np.where(trainOutNeuron == trainOutNeuron.max())
            
            #Increment count in confusion matrix
            confusion_train[expected_train_out_max_index[0][0]][train_out_max_index[0][0]] = int(confusion_train[expected_train_out_max_index[0][0]][train_out_max_index[0][0]] + 1)
            #Get next dp
            expected_output_index_train = expected_output_index_train + 1
        
        #Calculate Training and Test Errors
        
        correct_test_predictions = 0
        correct_train_predictions = 0
        for i in range (10):
            correct_train_predictions = correct_train_predictions + confusion_train[i][i]
            correct_test_predictions = correct_test_predictions + confusion_test[i][i]
        
        trainEF = (total_train_inputs - correct_train_predictions)/total_train_inputs
        testEF = (total_test_inputs - correct_test_predictions)/total_test_inputs
        
        #Return confusion matrix
        conf_train = pd.DataFrame(confusion_train)
        conf_test = pd.DataFrame(confusion_test)
        
        return (conf_train,conf_test,trainEF,testEF)
        
    def feedForward_backPropogate(self,full_df,epoch):
        #Save training error for epochs
        trainErrorFrac = []
        #Save testing error for epochs
        testErrorFrac = []
        #All epoch outputs
        epoch_outputs = []
        low = 0.25
        high = 0.75
        
        for e in range(epoch):
            #Get random Test and Train data points
            (X_traint,Y_traint,X_testt,Y_testt) = getRandomTrainTestSet(full_df,0.8)
            
            testErrors = 0
            trainErrors = 0

            XY_traint = pd.concat([X_traint, Y_traint], axis=1, join='inner')
            XY_testt = pd.concat([X_testt, Y_testt], axis=1, join='inner')
            
            #Train using random 25% of inputs in every epoch
            XY_traint = XY_traint.sample(frac=0.25,axis=0)
            XY_testt = XY_testt.sample(frac=1,axis=0)

            X_train_newt = XY_traint.drop(['label'],inplace=False,axis=1).reset_index(drop=True)
            Y_train_newt = XY_traint['label'].reset_index(drop=True)
            
            X_test_newt = XY_testt.drop(['label'],inplace=False,axis=1).reset_index(drop=True)
            Y_test_newt = XY_testt['label'].reset_index(drop=True)
            
            X_train_matt = X_train_newt.to_numpy()
            Y_train_matt = Y_train_newt.to_numpy()

            X_test_matt = X_test_newt.to_numpy()
            Y_test_matt = Y_test_newt.to_numpy()

            expected_output_index_train = 0 #For getting expected(true) output for Train
            expected_output_index_test = 0 #For getting expected(true) output for Test

            total_train_inputs = len(X_train_matt)
            total_test_inputs = len(X_test_matt)
            
            print(f'*** Running EPOCH:{e+1} ***')
            start_time = time.time()
        
            #Train & Test Simulation
            for index in range(len(X_train_matt)): #For every train data point
                correctPrediction = True
                correctPredictionTest = True
                
                layer_outputs = []
                layer_outputs_test = []
                
                data_point_test = X_test_matt[index]
                data_point = X_train_matt[index]
                
                #Test Predictions - WINNER TAKE All
                for layer in range(self.total_layers): #For each layer 128=>64=>10
                    output_data_point_test = []
                    if layer != output_layer_index: #Hidden Layer 128=>64
                        for neuron in self.neuralNetwork[layer]: #Every neuron in current layer
                            if layer == 0: #First hidden layer
                                neuron.setCurrentInput(data_point_test)
                            else: #Other hidden layers
                                neuron.setCurrentInput(layer_outputs_test[layer-1])
                            output_data_point_test.append(neuron.getCurrentOutput())
                    else: #Output Layer 10
                        for neuron in self.neuralNetwork[layer]: #Every neuron in current layer
                            neuron.setCurrentInput(layer_outputs_test[layer-1])
                            output_data_point_test.append(neuron.getCurrentOutput())
                    layer_outputs_test.append(np.array(output_data_point_test))
                    
                #Compare index of max value for both
                expected_test = Y_test_matt[expected_output_index_test] #Get expected output of data point
                expected_output_layer_test = y_ideal.get(expected_test) #Get expected output for all 10 neurons that make up the digit
                expected_output_layer_test = np.array(expected_output_layer_test)
                expected_test_out_max_index = np.where(expected_output_layer_test == expected_output_layer_test.max())
                
                testOutNeuron = layer_outputs_test[-1] #Output layer results
                test_out_max_index = np.where(testOutNeuron == testOutNeuron.max())
                
                if test_out_max_index[0][0] != expected_test_out_max_index[0][0]:
                    correctPredictionTest = False
                    
                if correctPredictionTest == False:
                    testErrors = testErrors + 1 #Update testing error
                
                #Feed forward Train
                for layer in range(self.total_layers): #For each layer 128=>64=>10
                    output_data_point = []
                    if layer != output_layer_index: #Hidden Layer 128=>64
                        for neuron in self.neuralNetwork[layer]: #Every neuron in current layer
                            if layer == 0: #First hidden layer
                                neuron.setCurrentInput(data_point)
                            else: #Other hidden layers
                                neuron.setCurrentInput(layer_outputs[layer-1])
                            output_data_point.append(neuron.getCurrentOutput())
                    else: #Output Layer 10
                        expected = Y_train_matt[expected_output_index_train] #Get expected output of data point
                        expected_output_layer = y_ideal.get(expected) #Get expected output for all 10 neurons that make up the digit
                        o = 0
                        for neuron in self.neuralNetwork[layer]: #Every neuron in current layer
                            output_delta = 0
                            neuron.setCurrentInput(layer_outputs[layer-1])
                            output_data_point.append(neuron.getCurrentOutput())
                            expected_value = expected_output_layer[o] #Expected
                            predicted_value = neuron.getStandardizedOutput(expected_value,low,high) #Predicted
                            output_delta = neuron.getOutputDelta() #Output neuron delta
                            if predicted_value != expected_value:
                                #Update weight
                                neuron.updateWeights()
                                correctPrediction = False
                            neuron.calculateWeightedDelta() #Populate weighted delta of output neurons for back propogation
                            o = o + 1 #Get expected output for next neuron in output layer
                    layer_outputs.append(np.array(output_data_point))
                
                if correctPrediction == False:
                    trainErrors = trainErrors + 1 #Update training error
                
                #Back propogate
                for layer in range(output_layer_index,0,-1): #From top layer 10->64->128
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

                expected_output_index_train = expected_output_index_train + 1
                expected_output_index_test = expected_output_index_test + 1
                        
            trainErrorFrac.append(trainErrors/total_train_inputs)
            testErrorFrac.append(testErrors/total_test_inputs)
            
            print("--- Execution: %s seconds ---" % (time.time() - start_time))
            
            #Break if training error fraction for current epoch reaches below 0.1
            if trainErrorFrac[-1] <= 0.001 and testErrorFrac[-1] <= 0.001:
                print('Stop Training - Training and Testing error fractions below 0.001')
                break
            
        return (trainErrorFrac,testErrorFrac)

#Function to plot error fraction for Train & Test
def plotErrorFraction(trainEF,testEF):
    
    E = [i+1 for i in range(len(trainEF))]
    fig,axes = plt.subplots(figsize=(10,5), num='Error Fraction')
    
    axes.set_title('Figure 3 Error Fraction vs Epoch')
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Error Fraction')

    axes.plot(E,testEF,label = 'Testing Set')
    axes.plot(E,trainEF,label = 'Training Set')
    
    axes.legend()
    fig.show()

#Display challenege matrix
def displayConfusionMatrix(data,err,title,num):
    plt.figure(num='Confusion Matrix')
    sns.heatmap(data, annot=True,cbar=False, cmap='coolwarm',fmt='g')
    plt.title(f'Figure {num} {title} Error Fraction:{err} with Confusion Matrix')
    plt.xticks(ticks=[0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5],labels=['0','1','2','3','4','5','6','7','8','9'])
    plt.yticks(ticks=[0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5],labels=['0','1','2','3','4','5','6','7','8','9'])
    plt.show()

#Create NeuralNetwork
backpropnn = NeuralNetwork(layers,layer_sizes,0.1,0.1,getSigmoid,getSignmodDerivative)

#Prediction before training
(conf_train,conf_test,trainEF,testEF)  = backpropnn.predict(XM.copy())

displayConfusionMatrix(conf_train,trainEF,'Before Training - Training Set',1)

displayConfusionMatrix(conf_test,testEF,'Before Training - Testing Set',2)

#Train Network
epoch = 400
(trainErrorFrac,testErrorFrac)  = backpropnn.feedForward_backPropogate(XM.copy(),epoch)

plotErrorFraction(trainErrorFrac,testErrorFrac)

#After training
(conf_train,conf_test,trainEF,testEF)  = backpropnn.predict(XM.copy())

#Display Train Confusion Matrix
displayConfusionMatrix(conf_train,trainEF,'After Training - Training Set',4)

#Display Test Confusion Matrix
displayConfusionMatrix(conf_test,testEF,'After Training - Testing Set',5)

print(f'Minimum Training Error Fraction:{min(trainErrorFrac)}') 
#[Minimum Training Error Fraction] => epoch,neurons,eta,alpha
#[0.312] => 30,50,0.1,0.1
#[0.243] => 50,50,0.1,0.1
#[0.209] => 70,50,0.1,0.1
#[0.154] => 100,50,0.1,0.1
#[0.126] => 120,50,0.1,0.1
#[0.086] => 150,50,0.1,0.1
#[0.071] => 175,50,0.1,0.1
#[0.058] => 200,50,0.1,0.1
#[0.027] => 250,50,0.1,0.1
#[0.006] => 300,50,0.1,0.1
#[0.007] => 400,50,0.1,0.1

print(f'Minimum Testing Error Fraction:{min(testErrorFrac)}') 
#[Minimum Testing Error Fraction] => epoch,neurons,eta,alpha
#[0.049] => 30,50,0.1,0.1
#[0.022] => 50,50,0.1,0.1
#[0.015] => 70,50,0.1,0.1
#[0.004] => 100,50,0.1,0.1
#[0.004] => 120,50,0.1,0.1
#[0.003] => 150,50,0.1,0.1
#[0.001] => 175,50,0.1,0.1
#[0.001] => 200,50,0.1,0.1
#[0.0] => 250,50,0.1,0.1
#[0.0] => 300,50,0.1,0.1
#[0.0] => 400,50,0.1,0.1