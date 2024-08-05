import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("D:\CODING\AIML\learnings\Regression Model\Polynomial linear Regression\\bmi_calculator\data.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


# Creating the linear regression model:

from sklearn.linear_model import LinearRegression

lin_reg= LinearRegression()
lin_reg.fit(x,y)


# Creating the polynomial regression model

from sklearn.preprocessing import PolynomialFeatures

poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)
print(x_poly)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)


wt=int(input("Enter the weight:\t"))
ht=float(input("Enter the height:\t"))

lin_res=lin_reg.predict([[wt ,ht]])

poly_res=lin_reg2.predict(poly_reg.fit_transform([[wt,ht]]))

print("The bmi calculated through multiple linear regression:\t"+str(lin_res))
print("The bmi calculated through polynomial linear regression:\t"+str(poly_res))


plt.scatter(x,y,color="blue")
plt.plot(x,poly_reg.fit_transform(x),color="black")
plt.ylabel("Salary->")
plt.xlabel("Level->")
plt.show()


