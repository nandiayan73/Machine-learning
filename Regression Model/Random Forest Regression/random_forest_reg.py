import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Position_salaries.csv")

x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

# Training the regression model:

from sklearn.ensemble import RandomForestRegressor
# n_estimators=no of trees,

# here we make a lot of tress by randomly selecting the data from the dataset,
# For a  value of data in which  we have to predict the output(y), we take the average of the result for the input from the deifferent trees.
# This is a very acuurate way to get the data.


regressor=RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(x,y)
print(regressor.predict([[7]]))


# Visualising the data:
X_grid = np.arange(min(x), max(x), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()