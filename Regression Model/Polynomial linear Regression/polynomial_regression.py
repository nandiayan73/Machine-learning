import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("D:\CODING\AIML\learnings\Regression Model\Polynomial linear Regression\Position_salaries.csv")
x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

# # splitting the dataset into train and test set
# from sklearn.



# Training the multiple linear regression model on the whole dataset:
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)


# Training the Polunomial regression model

# Importing the library
from sklearn.preprocessing import PolynomialFeatures

#Creating a matrix of squared forms
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)#x_poly is the matrix of squared forms of the values of x.
print(x_poly)

# Generating a linear regression model from the x_poly.
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)




# Visualising the data:
plt.scatter(x,y,color="red")
plt.xlabel("Level")
plt.ylabel("Salary")
# plt.plot(x,lin_reg.predict(x),color="black")
plt.show()

# Visualising the linear regression
plt.scatter(x,y,color="red")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.plot(x,lin_reg.predict(x),color="black")
plt.show()


# visualising the polynomial regression
plt.scatter(x,y,color="black")
plt.xlabel("Level->")
plt.ylabel("Salary->")
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color="purple")
plt.show()

n=float(input("Enter the level:\t"))
print(lin_reg.predict([[n]]))
print(lin_reg2.predict(poly_reg.fit_transform([[n]])))




# n = number of data points = 10

# Sum of x = 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 = 55
# Sum of x_squared = 1 + 4 + 9 + 16 + 25 + 36 + 49 + 64 + 81 + 100 = 385
# Sum of x_constant = 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 = 10
# Sum of y = 45000 + 50000 + 60000 + 80000 + 110000 + 150000 + 200000 + 300000 + 500000 + 1000000 = 3150000
# Sum of x*y = (1*45000) + (2*50000) + (3*60000) + (4*80000) + (5*110000) + (6*150000) + (7*200000) + (8*300000) + (9*500000) + (10*1000000) = 18450000

# Using the formulas for b0, b1, and b2:
# b1 = (n * sum of x*y - sum of x * sum of y) / (n * sum of x_squared - (sum of x)^2)
# b2 = (sum of y - b0 * sum of x_constant - b1 * sum of x) / n

# Substituting the values:
# b1 = (10 * 18450000 - 55 * 3150000) / (10 * 385 - 55^2)
# b2 = (3150000 - b0 * 10 - b1 * 55) / 10

# Now, we need to solve these equations simultaneously to find the value of b0.