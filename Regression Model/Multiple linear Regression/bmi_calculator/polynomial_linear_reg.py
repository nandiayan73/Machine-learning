import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("data.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values




# # splitting the dataset into train and test set


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=0)


# Training the multiple linear regression model on the whole dataset:
from sklearn.linear_model import LinearRegression


# Training the Polynomial regression model

# Importing the library
from sklearn.preprocessing import PolynomialFeatures

#Creating a matrix of squared forms
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x_train)#x_poly is the matrix of squared forms of the values of x.

# Generating a polynomial linear regression model from the x_poly.
lin_reg=LinearRegression()
lin_reg.fit(x_poly,y_train)


y_pred=lin_reg.predict(poly_reg.fit_transform(x_test))

from sklearn.metrics import r2_score

score=r2_score(y_test,y_pred)
print(score)

weight=float(input("Enter your weight:"))
height=float(input("Enter your height:"))
x_temp=np.array([[weight, height]])
bmi=lin_reg.predict(poly_reg.fit_transform(x_temp))
print(bmi)


# Score=0.9999102674932828




