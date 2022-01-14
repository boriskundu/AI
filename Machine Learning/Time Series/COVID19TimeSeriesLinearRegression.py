# Author: BORIS KUNDU
# A kind of "autoregression" by using the same time series data as both input and output.
# Uses the US Covid-19 hospitalizations data from hospitalizations.txt
# A random stretch of m = 60 days for training is used.
# Testing immediately after training and once also in a random future.
# Displayed mean-squared-errors.
# Actual and predicted hospitalizations plotted.
# The time-series window is n = 30 but it can be changed to find the best one.

#Import packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Read data
data = np.genfromtxt('hospitalizations.txt', delimiter=' ')
hosp = data[:,1]

#Plot data
plt.plot(hosp)
plt.title('US Covid Hospitalization data')
plt.xlabel('Data Points (Continuous Days)')
plt.ylabel('Number of Hospitalizations')
plt.show()

last = len(hosp) # Total data points
win = 30 # input size or window size
m = 60 # data period and number of data points

#Get start index of Training set
trainbegin = np.random.randint(400)
#Create one training set
Xtrain = np.array(hosp[trainbegin:trainbegin+win])
ytrain = np.array(hosp[trainbegin+win])

#Create 49+1 sets of overlapping training sets
for i in range(1, m):
    Xtrain = np.vstack([Xtrain, hosp[trainbegin+i:trainbegin+i+win]])
    ytrain = np.append(ytrain, hosp[trainbegin+i+win])

#Train LinearRegression model
lin_reg = LinearRegression()
lin_reg.fit(Xtrain, ytrain)
#Predict on training set.
ypred = lin_reg.predict(Xtrain)

print('Mean Training Error:', np.sqrt(mean_squared_error(ytrain, ypred)))

#Plot results
plt.plot(ytrain)
plt.plot(ypred)
plt.legend(['True hospitalizations','Predicted hospitalizations'])
plt.title('Training began at ' + str(trainbegin) + ' for window (number of days) ' + str(win))
plt.ylabel('Number of Hospitalizations')
plt.xlabel('Data Points (Continuous Days)')
plt.show()

#Initialize start index of testing set
testbegin = trainbegin + m
#Create one testing set
Xtest = np.array(hosp[testbegin:testbegin+win])
ytest = np.array(hosp[testbegin+win])

#Create 4+1 sets of overlapping testing sets
for i in range(1, 5):
    Xtest = np.vstack([Xtest, hosp[testbegin+i:testbegin+i+win]])
    ytest = np.append(ytest, hosp[testbegin+i+win])

#Predict on testing set
ypred = lin_reg.predict(Xtest)

print('Test error immediately after training:', np.sqrt(mean_squared_error(ytest, ypred)))

#Plot results
plt.plot(ytest)
plt.plot(ypred)
plt.legend(['True hospitalizations','Predicted hospitalizations'])
plt.title('Testing began at ' + str(testbegin) + ' for window (number of days) ' + str(win))
plt.ylabel('Number of Hospitalizations')
plt.xlabel('Data Points (Continuous Days)')
plt.show()

#Create random future testing set
testbegin = np.random.randint(trainbegin + 2 * m, last - win - m)
Xtest = np.array(hosp[testbegin:testbegin+win])
ytest = np.array(hosp[testbegin+win])
for i in range(1, m):
    Xtest = np.vstack([Xtest, hosp[testbegin+i:testbegin+i+win]])
    ytest = np.append(ytest, hosp[testbegin+i+win])

#Predict in future
ypred = lin_reg.predict(Xtest)
print('Test error at some random future period of time:', np.sqrt(mean_squared_error(ytest, ypred)))

#Plot reults
plt.plot(ytest)
plt.plot(ypred)
plt.legend(['True hospitalizations','Predicted hospitalizations'])
plt.title('Testing in the future began at ' + str(testbegin) + ' for window (number of days) ' + str(win))
plt.ylabel('Number of Hospitalizations')
plt.xlabel('Data Points (Continuous Days)')
plt.show()