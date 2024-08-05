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

y = y.reshape(len(y),1)

# Splitting the data into training and test sets:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)



# Feature Scaling:

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x_train[:,4:] = sc_x.fit_transform(x_train[:,4:])
x_test[:,4:] = sc_x.fit_transform(x_test[:,4:])
y_train = sc_y.fit_transform(y_train)



# Model Design:
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(x_train,y_train)

# Predicting the results:
scaled_res=regressor.predict(x_test)
y_pred=sc_y.inverse_transform(scaled_res.reshape(-1,1))

print(y_test)
print(y_pred)


# Evaluating the score:
from sklearn.metrics import r2_score

score=r2_score(y_test,y_pred)
print(score)