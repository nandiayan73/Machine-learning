import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("data.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

# *******Data Preprocessing********:
from sklearn.preprocessing import LabelEncoder



le2=LabelEncoder()
x[:,0]=le2.fit_transform(x[:,0])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct=ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[2])],remainder="passthrough")
x=np.array(ct.fit_transform(x))


# Splitting the data into training and test sets:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# Multiple Linear Regression Model Training:

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(x_train,y_train)

print(x_train)
# Evaluating the score:
y_pred=regressor.predict(x_test)
print(y_test)
print(y_pred)

from sklearn.metrics import r2_score

print(r2_score(y_test,y_pred))



