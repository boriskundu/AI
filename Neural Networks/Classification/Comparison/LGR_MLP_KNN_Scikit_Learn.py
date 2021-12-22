# Comparing various classification models using Scikit Learn on non-linearly separable data

#Impoer packages
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

features, labels= make_moons(n_samples=250)
plt.scatter(features.T[0,:],features.T[1,:],c=labels)
plt.title('Noiseless Data')
plt.show()

#Get noisy samples
features, labels= make_moons(n_samples=250, noise=0.1)
plt.figure()
plt.scatter(features.T[0,:],features.T[1,:],c=labels)
plt.title('Noisy Data')
plt.show()

# # Standardize Data

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
plt.figure()
plt.title('Noisy Data after Standardization')
plt.scatter(scaled_features.T[0,:],scaled_features.T[1,:],c=labels)
plt.show()

# ## Split Data

X_training, X_testing, Y_training, Y_testing = train_test_split(scaled_features, labels, test_size=0.3)

plt.figure()
plt.subplot(1,2,1)
plt.scatter(X_training.T[0,:],X_training.T[1,:],c=Y_training)
plt.title("Training Data")
plt.subplot(1,2,2)
plt.scatter(X_testing.T[0,:],X_testing.T[1,:],c=Y_testing)
plt.title("Test Data")
plt.show()

# # Logistic Regression

model = LogisticRegression()
model.fit(X_training,Y_training)
LGR_Predicted=model.predict(X_testing)

plt.figure()
plt.subplot(1,2,1)
plt.scatter(X_testing.T[0,:],X_testing.T[1,:],c=LGR_Predicted)
plt.title("LGR classified Data")
plt.subplot(1,2,2)
plt.scatter(X_testing.T[0,:],X_testing.T[1,:],c=Y_testing)
plt.title("True Labels - Test Data")
plt.show()

print('LGC result:')
print('Accuracy: ',accuracy_score(Y_testing,LGR_Predicted))
print('f1_score: ',f1_score(Y_testing,LGR_Predicted))
print('Precision: ', precision_score(Y_testing,LGR_Predicted))
print('Recall: ', recall_score(Y_testing,LGR_Predicted))


# # Classification using Neural Networks
# ## No hidden Layer

MLP = MLPClassifier(activation='logistic',solver='sgd',hidden_layer_sizes=(),max_iter=1000)
MLP.fit(X_training, Y_training)

#Make predictions
MLP_Predicted=MLP.predict(X_testing)

plt.figure()
plt.subplot(1,2,1)
plt.scatter(X_testing.T[0,:],X_testing.T[1,:],c=MLP_Predicted)
plt.title("MLP classified Data - No hidden layer")
plt.subplot(1,2,2)
plt.scatter(X_testing.T[0,:],X_testing.T[1,:],c=Y_testing)
plt.title("True Label - Test Data")
plt.tight_layout()
plt.show()

print('MLP result without hidden layer:')
print('Accuracy: ',accuracy_score(Y_testing,MLP_Predicted))
print('f1_score: ',f1_score(Y_testing,MLP_Predicted))
print('Precision: ', precision_score(Y_testing,MLP_Predicted))
print('Recall: ', recall_score(Y_testing,MLP_Predicted))

# ## Let's add hidden layer

MLP = MLPClassifier(activation='logistic',solver='lbfgs',hidden_layer_sizes=(5,),max_iter=5000)
#Solver=lbfgs
MLP.fit(X_training, Y_training)

#Predict
MLP_Predicted=MLP.predict(X_testing)

plt.figure()
plt.subplot(1,2,1)
plt.scatter(X_testing.T[0,:],X_testing.T[1,:],c=MLP_Predicted)
plt.title("MLP classified Data - With hidden layer")
plt.subplot(1,2,2)
plt.scatter(X_testing.T[0,:],X_testing.T[1,:],c=Y_testing)
plt.title("True Labels- Test Data")
plt.tight_layout()
plt.show()

print('MLP result with hidden layer:')
print('Accuracy: ',accuracy_score(Y_testing,MLP_Predicted))
print('f1_score: ',f1_score(Y_testing,MLP_Predicted))
print('Precision: ', precision_score(Y_testing,MLP_Predicted))
print('Recall: ', recall_score(Y_testing,MLP_Predicted))

# # KNN

KNN = KNeighborsClassifier(n_neighbors=5)
KNN.fit(X_training, Y_training)
KNN_Predicted=KNN.predict(X_testing)

plt.figure()
plt.subplot(1,2,1)
plt.scatter(X_testing.T[0,:],X_testing.T[1,:],c=KNN_Predicted)
plt.title("KNN classified Data")
plt.subplot(1,2,2)
plt.scatter(X_testing.T[0,:],X_testing.T[1,:],c=Y_testing)
plt.title("Test Data")
plt.show()

print('KNN result:')
print('Accuracy: ',accuracy_score(Y_testing,KNN_Predicted))
print('f1_score: ',f1_score(Y_testing,KNN_Predicted))
print('Precision: ', precision_score(Y_testing,KNN_Predicted))
print('Recall: ', recall_score(Y_testing,KNN_Predicted))