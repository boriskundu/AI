# # MDC and KNN Classification Result Evaluation 

#Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score

#Read data
data=pd.read_csv('StressDataset.txt',delimiter='\t')
data=data.rename(columns={'Wife Salary':'X', 'Husband Salary':'Y', 'Stressed':'L'})


# ## Visualize Data

#Plot data
plt.scatter(data['X'], data['Y'],c=data['L'])
plt.xlabel("Wife Salary ($100,000 units)")
plt.ylabel("Husband Salary ($100,000 units)")
plt.title('Stressed Dataset')
plt.show()


# ## Split Data
X_training, X_testing, Y_training, Y_testing = train_test_split(data[['X','Y']], data['L'], test_size=0.3)

plt.figure()
plt.scatter(X_training['X'][Y_training==0],X_training['Y'][Y_training==0],label='Not Stressed',color='blue')
plt.scatter(X_training['X'][Y_training==1],X_training['Y'][Y_training==1],label='Stressed',color='red')
plt.xlabel("Wife Salary ($100,000 units)")
plt.ylabel("Husband Salary ($100,000 units)")
plt.title('Stressed Dataset - Training')
plt.legend()
plt.show()
plt.figure()
plt.scatter(X_testing['X'][Y_testing==0],X_testing['Y'][Y_testing==0],label='Not Stressed',color='blue')
plt.scatter(X_testing['X'][Y_testing==1],X_testing['Y'][Y_testing==1],label='Stressed',color='red')
plt.xlabel("Wife Salary ($100,000 units)")
plt.ylabel("Husband Salary ($100,000 units)")
plt.title('Stressed Dataset - Test')
plt.legend()
plt.show()

# ## MDC Classifier
#Train model
model=NearestCentroid()
model.fit(X_training,Y_training)

#Predict
Y_training_predicted=model.predict(X_training)
Y_testing_predicted=model.predict(X_testing)

# ## K-Nearst Neighbors Classifier
modelKNN = KNeighborsClassifier(n_neighbors=5)
modelKNN.fit(X_training,Y_training)

Y_training_predicted_KNN=modelKNN.predict(X_training)
Y_testing_predicted_KNN=modelKNN.predict(X_testing)


# ## Visualize the decision boundary

x_grid=np.arange(0,15,0.1)
y_grid=np.arange(0,25,0.1)
data_grid=[[i,j] for j in y_grid for i in x_grid]
'''
xx,yy=np.meshgrid(x_grid,y_grid)
data_grid = np.array([xx, yy]).reshape(2, -1).T
'''
Predicted_MDC=model.predict(data_grid)
Predicted_KNN=modelKNN.predict(data_grid)


# In[ ]:





# In[ ]:





# In[56]:


plt.figure()
plt.contourf(x_grid,y_grid,Predicted_MDC.reshape(y_grid.shape[0],x_grid.shape[0]))
plt.scatter(X_testing['X'][Y_testing==0],X_testing['Y'][Y_testing==0],label='Not Stressed',color='blue')
plt.scatter(X_testing['X'][Y_testing==1],X_testing['Y'][Y_testing==1],label='Stressed',color='red')
plt.scatter(X_testing['X'][Y_testing!=Y_testing_predicted],X_testing['Y'][Y_testing!=Y_testing_predicted],label='Wrong classification',color='green')
plt.scatter(X_training['X'][Y_training==0].mean(),X_training['X'][Y_training==0].mean(),c='orange')
plt.scatter(X_training['X'][Y_training==1].mean(),X_training['X'][Y_training==1].mean(),c='orange')
plt.xlabel("Wife Salary ($100,000 units)")
plt.ylabel("Husband Salary ($100,000 units)")
plt.title('MDC Classification Result - Test Data')
plt.legend()
plt.show()

plt.figure()
plt.contourf(x_grid,y_grid,Predicted_KNN.reshape(y_grid.shape[0],x_grid.shape[0]))
plt.scatter(X_testing['X'][Y_testing==0],X_testing['Y'][Y_testing==0],label='Not Stressed',color='blue')
plt.scatter(X_testing['X'][Y_testing==1],X_testing['Y'][Y_testing==1],label='Stressed',color='red')
plt.scatter(X_testing['X'][Y_testing!=Y_testing_predicted_KNN],X_testing['Y'][Y_testing!=Y_testing_predicted_KNN],label='Wrong classification',color='green')
plt.xlabel("Wife Salary ($100,000 units)")
plt.ylabel("Husband Salary ($100,000 units)")
plt.title('KNN Classification Result - Test Data')
plt.legend()
plt.show()

#confusion_matrix(Y_testing,Y_testing_predicted)
plot_confusion_matrix(model,X_testing,Y_testing)
plt.title('MDC')
plt.show()

plot_confusion_matrix(modelKNN,X_testing,Y_testing)
plt.title('KNN')
plt.show()

print('MDC result:')
print(accuracy_score(Y_testing,Y_testing_predicted))
print(precision_score(Y_testing,Y_testing_predicted))
print(recall_score(Y_testing,Y_testing_predicted))

print('KNN result:')
print(accuracy_score(Y_testing,Y_testing_predicted_KNN))
print(precision_score(Y_testing,Y_testing_predicted_KNN))
print(recall_score(Y_testing,Y_testing_predicted_KNN))

# ### Visualise TP/TN/FP/FN

plt.figure()
plt.contourf(x_grid,y_grid,Predicted_MDC.reshape(y_grid.shape[0],x_grid.shape[0]))
#True Positive - Stressed and classified as stressed
plt.scatter(X_testing['X'][Y_testing+Y_testing_predicted==2],X_testing['Y'][Y_testing+Y_testing_predicted==2],label='Stressed (TP)',color='red')
#False Negative - Stressed and classified worng
plt.scatter(X_testing['X'][Y_testing>Y_testing_predicted],X_testing['Y'][Y_testing>Y_testing_predicted],label='Stressed but classified wrong (FN)',marker='+',color='red')
#True Negative - Not Stressed and classified as stressed
plt.scatter(X_testing['X'][Y_testing+Y_testing_predicted==0],X_testing['Y'][Y_testing+Y_testing_predicted==0],label='Not Stressed (TN)',color='green')
#False Positive - Not Stressed and classified wrong
plt.scatter(X_testing['X'][Y_testing<Y_testing_predicted],X_testing['Y'][Y_testing<Y_testing_predicted],label='Not Stressed but classified wrong (FP)',marker='+',color='green')
plt.xlabel("Wife Salary ($100,000 units)")
plt.ylabel("Husband Salary ($100,000 units)")
plt.title('Classification Result - MDC - Test Data')
plt.legend()
plt.show()

plt.figure()
plt.contourf(x_grid,y_grid,Predicted_KNN.reshape(y_grid.shape[0],x_grid.shape[0]))
#True Positive - Stressed and classified as stressed
plt.scatter(X_testing['X'][Y_testing+Y_testing_predicted_KNN==2],X_testing['Y'][Y_testing+Y_testing_predicted_KNN==2],label='Stressed (TP)',color='red')
#False Negative - Stressed and classified worng
plt.scatter(X_testing['X'][Y_testing>Y_testing_predicted_KNN],X_testing['Y'][Y_testing>Y_testing_predicted_KNN],label='Stressed but classified wrong (FN)',marker='+',color='red')
#True Negative - Not Stressed and classified as stressed
plt.scatter(X_testing['X'][Y_testing+Y_testing_predicted_KNN==0],X_testing['Y'][Y_testing+Y_testing_predicted_KNN==0],label='Not Stressed (TN)',color='green')
#False Positive - Not Stressed and classified wrong
plt.scatter(X_testing['X'][Y_testing<Y_testing_predicted_KNN],X_testing['Y'][Y_testing<Y_testing_predicted_KNN],label='Not Stressed but classified wrong (FP)',marker='+',color='green')
plt.xlabel("Wife Salary ($100,000 units)")
plt.ylabel("Husband Salary ($100,000 units)")
plt.title('Classification Result - KNN - Test Data')
plt.legend()
plt.show()