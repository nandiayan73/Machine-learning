import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("data.csv")

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=0)

from sklearn.ensemble import RandomForestRegressor



regressor=RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)


weight=float(input("Enter your weight:"))
height=float(input("Enter your height:"))
x_temp=np.array([[weight, height]])
bmi=regressor.predict(x_temp)
print(bmi)


from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print(score)