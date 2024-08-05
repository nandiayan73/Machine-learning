import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("D:\CODING\AIML\learnings\Regression Model\Regression Model Selection\evaluation and tests\Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing  import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x_train)

regressor=LinearRegression()
regressor.fit(x_poly,y_train)

y_pred = regressor.predict(poly_reg.transform(x_test))

# Evaluating the Model Performance:

from sklearn.metrics import r2_score

print(r2_score(y_test,y_pred))





