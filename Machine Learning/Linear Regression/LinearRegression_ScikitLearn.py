# # Linear Regression using Scikit Learn on Advertising data
# ## TotalSales = $w_{0}$ + $w_{1}$ * (TelevisionAd) + $w_{2}$ * $(TelevisionAd)^{2}$ + $w_{3}$ * (RadioAd) + $w_{4}$ * $(RadioAd)^{2}$
# ## Author: Boris Kundu

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ## Read CSV file and display head and info.

ads = pd.read_csv('Advertising.csv')
print(f'Info:\n{ads.info}')
print(f'\nHead:\n{ads.head()}')

# ## Add two new features to our dataset.

ads['TV_SQR'] = ads['TV']**2
ads['RADIO_SQR'] = ads['Radio']**2
print(f'\nCheck head after adding features:\n{ads.head()}')

# ## Add features in X and target feature in y

X = ads[['TV','TV_SQR','Radio','RADIO_SQR']]
y = ads['Sales']
m = len(ads['Sales'])

# ## Add features in X and target feature in y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# ## Create and Train model

model = LinearRegression(normalize=True)
model.fit(X_train,y_train)

# ## Display intercept (bias) and coefficients (feature weights)

print(f'Bias:{model.intercept_}')
print(f'Feature Weights:{model.coef_}')

# ## Predict sales on Test dataset

predicted_sales = model.predict(X_test)
predicted_sales_train = model.predict(X_train)

# ## Plot TV Ad expenditure vs Expected Sales and Predicted Sales.

plt.scatter(X_test['TV'],y_test,label='Expected')
plt.scatter(X_test['TV'],predicted_sales,label='Predicted')
plt.xlabel('TV Advertisement Expenditure')
plt.ylabel('Sales')
plt.legend()
plt.show()

# ## Plot Radio Ad expenditure vs Expected Sales and Predicted Sales.

plt.figure()
plt.scatter(X_test['Radio'],y_test,label='Expected')
plt.scatter(X_test['Radio'],predicted_sales,label='Predicted')
plt.xlabel('Radio Advertisement Expenditure')
plt.ylabel('Sales')
plt.legend()
plt.show()

# ## Get Mean Squared Error for Train & Test sets.

MSE_Test = mean_squared_error(y_test,predicted_sales)
print (f'MSE for Test is:{MSE_Test}')
MSE_Train = mean_squared_error(y_train,predicted_sales_train)
print (f'MSE for Train is:{MSE_Train}')