## BackPropogation using Sigmoid and Hyperbolic Tangent activation functions for Banknote dataset
#### Authors - Boris Kundu

#Import packages
import numpy as np
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import copy
from sklearn.model_selection import train_test_split
from random import seed
from random import random
from random import randrange

"""## Reading CSV Data File"""

#Read data and add feature names
dataset = pd.read_csv('data_banknote_authentication.txt',names=['variance','skewness','curtosis','entropy','class'])

"""### Printing Dataset Information"""
print('='*100)
print("Visualizing Dataset: ")
#Check info
print(f'Info:\n{dataset.info}')

#Check header
print(f'Head:\n{dataset.head()}')

"""## Data Preprocessing

Involes renaming columns and normalizing the data.
"""

# Data-preprocessing
#Define training features in X
X = dataset[['variance','skewness','curtosis','entropy']]
#Define output target feature in Y
Y = dataset[['class']]

#Standardize numerical features of X
X = (X-np.mean(X))/np.std(X)

#Add bias feature in X
#X['bias'] = np.ones(len(Y))

"""## Generating Datasets for Training, Testing and Validation"""

#Create Train (including validate) and Test sets
X_train_full, X_test, Y_train_full, Y_test = train_test_split(X, Y, train_size=0.8, random_state=101)
#Create Train and Validate sets
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_full, Y_train_full, train_size=0.2, random_state=101)

total_features = len(X_train.columns)

# Calculate weighted sum of input features for a neuron
def getWeightedSum(weights, inputs):
	weightedSum = weights[-1]
	for i in range(len(weights)-1):
		weightedSum += weights[i] * inputs[i]
	return weightedSum

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Signmoid activation
def getSigmoid(weightedSum):
	return 1.0 / (1.0 + np.exp(-weightedSum))

# Sigmoid derivative
def getSigmoidDerivative(sigmoid_output):
	return sigmoid_output * (1.0 - sigmoid_output)

# Hyperbolic tangent Activation
def getHyperTangent(weightedSum):
  return (np.exp(weightedSum) - np.exp(-weightedSum))/(np.exp(weightedSum) + np.exp(-weightedSum))

# Hyperbolic tangent Derivative
def getHypTanDerivative(tanh_output):
  return 1.0 - tanh_output * tanh_output

#Defining Threshold Function for changing threshold, based on activation function
def define_threshold(activation_func):
  act_func = str(activation_func)
  l, h = None, None
  if 'getSigmoid' in act_func:
    l, h= 0.25, 0.75
  elif 'getHyperTangent' in act_func:
    l, h = -0.5, 0.5
  else:
    l, h = None, None
  return l, h

#Defining output labels, based on the activation function
def define_output(act_function):
  act_func = str(act_function)
  l, h = None, None
  if 'getSigmoid' in act_func:
    l, h= 0, 1
  elif 'getHyperTangent' in act_func:
    l, h = -1, 1
  else:
    l, h = None, None
  return l, h

# Forward propagate input to a network output
def forward_propagate(network, feature_inputs,activation_function=getSigmoid):
	inputs = feature_inputs
	l, h = define_threshold(activation_function)
	l_out, h_out = define_output(activation_function)
	for layer in network:
		new_inputs = []
		for neuron in layer:
			weightedSum = getWeightedSum(neuron['weights'], inputs)
			neuron['output'] = activation_function(weightedSum)
			if neuron['output'] >= h:
				neuron['output'] = h_out
			elif neuron['output'] <= l:
				neuron['output'] = l_out
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Initialize Weights
def getInitialWeights(feature_count):
    limit = np.sqrt(3/feature_count)
    initialWeights = np.random.uniform(-limit, limit,feature_count)
    return initialWeights

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected, activation_derivative_function=getSigmoidDerivative):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(neuron['output'] - expected)
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * activation_derivative_function(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] -= l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, x, y, eta=0.1, epochs=50, activation_function=getSigmoid, activation_derivative_function=getSigmoidDerivative):
	for epoch in range(epochs):
		for data_point, expected in zip(x, y):
			outputs = forward_propagate(network, data_point, activation_function)
			backward_propagate_error(network, expected, activation_derivative_function)
			update_weights(network, data_point, eta)

# Initialize a network
def initialize_network(n_inputs=total_features, n_hidden=total_features, n_outputs=1, init_weights=None):
	network = list()
	if init_weights:
		hidden_layer = [{'weights': init_weights[0][i]} for i in range(n_hidden)]	
		output_layer = [{'weights': init_weights[1][i]} for i in range(n_outputs)]	
	else:
		hidden_layer = [{'weights':getInitialWeights(n_inputs+1)} for i in range(n_hidden)]
		output_layer = [{'weights':getInitialWeights(n_hidden+1)} for i in range(n_outputs)]
	network.append(hidden_layer)
	network.append(output_layer)
	return network

# Make a prediction with a network
def predict(network, row, activation_function):
	outputs = forward_propagate(network, row,activation_function)
	return outputs[-1]

# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(x_train,y_train,x_test,y_test, eta=0.1, epochs=50, 
                     n_hidden = total_features,
										 activation_function=getSigmoid,activation_derivative_function=getSigmoidDerivative, 
										 init_weights=None):
	n_inputs = len(x_train[0])
	n_outputs = len(y_train[0])
	network = initialize_network(n_inputs, n_hidden, n_outputs, init_weights)
	train_network(network,x_train,y_train,eta,epochs,activation_function,activation_derivative_function)
	predictions = list()
	for row in x_test:
		prediction = predict(network, row,activation_function)
		predictions.append(prediction)	
	return(accuracy_metric(y_test,predictions), network)

"""#### Defining X, Y Matrices"""

#Initializing matrices of the training, testing and validation sets
X_train_mat = X_train.to_numpy()
Y_train_mat = Y_train.to_numpy()

X_test_mat = X_test.to_numpy()
Y_test_mat = Y_test.to_numpy()

X_valid_mat = X_valid.to_numpy()
Y_valid_mat = Y_valid.to_numpy()

#Transformation in data for Tanh Function
Y_train_tanh = Y_train.copy()
Y_train_tanh['class'] = Y_train_tanh['class'].apply(lambda y: -1 if y == 0 else 1)
Y_valid_tanh = Y_valid.copy()
Y_valid_tanh['class'] = Y_valid_tanh['class'].apply(lambda y: -1 if y == 0 else 1)
Y_test_tanh = Y_test.copy()
Y_test_tanh['class'] = Y_test_tanh['class'].apply(lambda y: -1 if y == 0 else 1)
Y_train_mat_tanh = Y_train_tanh.to_numpy()
Y_valid_mat_tanh = Y_valid_tanh.to_numpy()
Y_test_mat_tanh = Y_test_tanh.to_numpy()

"""#### Initializing new weights"""

# Function to get initialiized weights for comparison
def getInitializedWeights(n_inputs=total_features, n_hidden=total_features, n_outputs=1):
  weights = []
  # Initializing Weights for hidden layer
  weights.append([getInitialWeights(n_inputs+1) for i in range(n_hidden)])
  weights.append([getInitialWeights(n_hidden+1) for i in range(n_outputs)])
  return weights

"""### Testing for best squashing Function

Testing for Activation Functions:

1.   Sigmoid Function
2.   Hyperbolic Tangent Function


"""
print('='*100)
print("Testing for the Best Squashing Function:\n")
# Setup
epochs = 300
n_hidden = 4

func = [("'Sigmoid'", getSigmoid, getSigmoidDerivative, Y_train_mat, Y_valid_mat), ("'Tanh'",getHyperTangent, getHypTanDerivative, Y_train_mat_tanh, Y_valid_mat_tanh)]
#Run on Validation set with different activation functions to get the best function
best_scores = { 'Eta': [], 'Sigmoid': [], 'Tanh': [], 'best_func': []}
eta_s = [0.01,0.02,0.03,0.04,0.05]
for eta in eta_s:
  print(f"Result with Eta {eta}:")
  scores = [] 
  for name, activation, activation_derivative, y_t, y_v in func:
    score = back_propagation(X_train_mat,y_t,X_valid_mat,y_v, eta, epochs, n_hidden,activation, activation_derivative)[0]
    scores.append(score)
    print(f'\tAccuracy of {name.rjust(10," ")} activation function with {n_hidden} hidden neurons is {score}%')

  # Evaluating the best activation function 
  max_score = max(scores)
  best_func = func[scores.index(max_score)]
  # print(f'\nThe activation function {best_func[0]} has the best score: {max_score}%')
  best_scores['Eta'].append(eta)
  best_scores['Sigmoid'].append(scores[0])
  best_scores['Tanh'].append(scores[1])
  best_scores['best_func'].append(best_func)

sig_scores = []
tanh_scores= []
for tanh_score, sig_score in zip(best_scores['Tanh'], best_scores['Sigmoid']):
  sig_scores.append(sig_score)
  tanh_scores.append(tanh_score)

plt.title("Learning Rate vs Accuracy")
plt.plot(eta_s, sig_scores,  label ="Sigmoid")
plt.plot(eta_s, tanh_scores,  label ="Hyperbolic Tangent")
plt.xlabel("Learning Rate (eta)")
plt.ylabel("Accuracy (%)")
plt.legend(loc="lower right")
plt.show()

## Evaluating Squashing Function
# Taking Max out of DataFrames
scores_df = pd.DataFrame(best_scores)
max_scores_df = scores_df[['Sigmoid','Tanh']].max()

if max_scores_df['Sigmoid'] > max_scores_df['Tanh']:
  max_df = scores_df[scores_df['Sigmoid'] == max_scores_df['Sigmoid']]
  best_eta = float(max_df['Eta'])
  best_squashing_func = (getSigmoid, getSigmoidDerivative)
  print(f'\nThe best Squashing Function is Sigmoid')  
else:
  max_df = scores_df[scores_df['Tanh'] == max_scores_df['Tanh']]
  best_eta = float(max_df['Eta'])
  best_squashing_func = (getHyperTangent, getHypTanDerivative)
  print(f'\nThe best Squashing Function is Tanh')

print(f"The best 'eta' is {best_eta}")
print('='*100)
print("Testing for the optimal number of neurons in the hidden layer\n")
# Setup
eta = best_eta

# Selecting best activation function
activation = best_squashing_func[0]
act_derivative = best_squashing_func[1]

#Run on Validation set with different hidden neurons till 1 and get best accuracy
scores = { 'hidden_neurons': [], 'score': [], 'net': []}
for n_hidden in range(4, 0,-1):
  weights = getInitializedWeights(len(X_train_mat[0]), n_hidden, len(Y_train_mat_tanh[0]))
  results = back_propagation(X_train_mat, Y_train_mat_tanh, X_valid_mat, Y_valid_mat_tanh, eta, epochs, n_hidden, activation, act_derivative, weights)
  scores['hidden_neurons'].append(n_hidden)
  scores['score'].append(results[0])
  scores['net'].append(results[1])
  print(f'Accuracy with {n_hidden} hidden neurons is {results[0]}%')

"""#### Selecting the best Scores"""

scores_df = pd.DataFrame(scores)
best_scores_df = scores_df[scores_df['score'] == scores_df['score'].max()]
best_hidden_neurons = int(best_scores_df.iloc[0]['hidden_neurons'])
best_network = best_scores_df.iloc[0]['net']
print(f"The best number of neurons for the hidden layer is '{best_hidden_neurons}'")

"""#### Train the network on best parameters found.

"""
print('='*100)
print("Testing the network on the best parameters found, on the testing set.\n")
print("Testing Dataset:")
print(X_test)

parameters = f"""\nParameters for 'Testing':
                 1. Best Learning Rate: {best_eta}
                 2. Epochs: {epochs}
                 3. Number of neurons in the hidden layer: {best_hidden_neurons}
                 4. Best Squashing Function: {best_squashing_func[0]}"""

print(parameters)
# Final Parameters
eta = best_eta
n_hidden = best_hidden_neurons

# Selecting best activation function
activation = best_squashing_func[0]
act_derivative = best_squashing_func[1]

#Test on best neural network
predictions = list()
for row in X_test_mat:
        prediction = predict(best_network, row, activation)
        predictions.append(prediction)
score = accuracy_metric(Y_test_mat_tanh,predictions)

print(f'Accuracy with {n_hidden} hidden neurons is {score}%')