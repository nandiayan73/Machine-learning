import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# replacing the missing data
from sklearn.impute import SimpleImputer

# encoding the categorical data
    # Changing the country into  vectors
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
    # Changing the yes/no to 1/0:
from sklearn.preprocessing import LabelEncoder


# splitting the dataset into train and test set
from sklearn.model_selection import train_test_split

# feature scalling

from sklearn.preprocessing import StandardScaler


dataset=pd.read_csv("D:\CODING\AIML\learnings\data preprocessing\Data.csv")
x=dataset.iloc[:,:-1].values


y=dataset.iloc[:,-1].values
# print(x)


# replacing the missing data
imputer=SimpleImputer(missing_values=np.nan,strategy="mean")
imputer.fit(x[:,1:3])#only the 2nd and 3rd column is selected for change
x[:,1:3]=imputer.transform(x[:,1:3]) #replacing only the 2nd and 3rd column in the array

# Encoding the categorical data:

    # Changing the country into  vectors

ct=ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[0])],remainder="passthrough")
temp=x
x=np.array(ct.fit_transform(x))
print(x)


    # Change yes/no to 1/0

le=LabelEncoder()
y=le.fit_transform(y)
# [0 1 0 0 1 1 0 1 0 1]


# splitting the dataset into train and test set

x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
print(x_train)
# print(x_test)
# print(y_test)


# Feature Scaling: Standardization
# we will be feature scaling only the portion with numerical values.
sc=StandardScaler()
x_train[:,3:]=sc.fit_transform(x_train[:,3:])
x_test[:,3:]=sc.transform(x_test[:,3:])


print(x_test)



