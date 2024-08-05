# Importing the librearies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset=pd.read_csv("Position_Salaries.csv")

test_dataset=pd.read_csv("data.csv")


print(test_dataset)



x2=test_dataset.iloc[:,1:-1].values
y2=test_dataset.iloc[:,-1].values

x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values


# Training the dataset
# Decision tree regression is not well adapted to only one independent variable column.

from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state = 0)
regressor.fit(x,y)

#The dataset will be split into several groups. The res of any input or level will be the average of the data present in a group of the splitted group,
# this is the reason why there will be same average value for a certain range of data


print(regressor.predict([[6.5]]))

# Visualising the result:
X_grid = np.arange(min(x), max(x), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()























































































































































































