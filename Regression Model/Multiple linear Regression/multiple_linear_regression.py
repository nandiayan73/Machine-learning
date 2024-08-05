import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importign the dataset
dataset=pd.read_csv("D:\CODING\AIML\learnings\Regression Model\Multiple linear Regression\\50_Startups.csv")
x=dataset.iloc[:,:-1].values#Everything except the last column
y=dataset.iloc[:,-1].values#THe dependent profits


# Encoding the categorical data: changign the categorical data into vectors

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
 
# Splitting the dataset into train and test set:

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)



# Multiple Linear Regression: Making the model

from sklearn.linear_model import LinearRegression
# The linear Regression will :
# 1.Take care of the dummy variable trap,
# 2. Automatically select the best model(backward elimination with the highest p-value)

# The p-values associated with the coefficients indicate the probability 
# that the predictor's coefficient is zero (i.e., it has no effect on the
# dependent variable).

regressor=LinearRegression()
regressor.fit(x_train,y_train)

#Predicting the test set result

y_pred=regressor.predict(x_test)
np.set_printoptions(precision=2)#only upto 2 decimal places will be selected
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))



