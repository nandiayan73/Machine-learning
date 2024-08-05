import numpy as np
import pandas as pd

dataset=pd.read_csv("D:\CODING\AIML\learnings\Regression Model\Multiple linear Regression\\bmi_calculator\data.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=0)


from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)
np.set_printoptions(precision=2)#only upto 2 decimal places will be selected
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))






weight=float(input("Enter your weight:"))
height=float(input("Enter your height:"))
x_temp=np.array([[weight, height]])
print(x_temp)
bmi=regressor.predict(x_temp)
print(bmi)


from sklearn.metrics import r2_score 
print(r2_score(y_test,y_pred))

# Score=0.912689631023624