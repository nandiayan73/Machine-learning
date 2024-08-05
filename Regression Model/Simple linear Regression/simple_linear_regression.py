# Data Preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
# depending on the years worked the salary will be decided.
# so salary is the dependent variable
# year is the feature
dataset=pd.read_csv("D:\CODING\AIML\learnings\Regression Model\Simple linear Regression\Salary_data.csv")
x=dataset.iloc[:,:-1].values #Except the last column takign everything
y=dataset.iloc[:,-1].values


# Splititng the dataset into test and train set:
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)


# Training the SIMPLE LINEAR REGRESSION MODEL ON THE TRAINING SET

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
# regressor.fir(the yrs of experience, the resultant salary)




# implementing the model on the test set
y_pred=regressor.predict(x_test)

# We are only entering the yrsExp. as we want to know the salary.




# Visualizing the train st results
plt.scatter(x_train,y_train,color="pink")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Salary vs experience(Training Set)")
plt.xlabel("yrs of exp->")
plt.ylabel("salary of exp(training set)")
plt.show()

# Visualizing the train set results
plt.scatter(x_test,y_test,color="pink")
# plt.plot(x_test,regressor.predict(x_test),color="black")#applying the test set regression model
plt.plot(x_train,regressor.predict(x_train),color="blue")#applying the training set regression model
plt.scatter(x_test,y_test,color="violet")
plt.title("Salary vs experience(Test Set)")
plt.xlabel("yrs of exp->")
plt.ylabel("salary of exp(test set)")
plt.show()



yrs=int(input("enter the years of experience:\t"))
temp=np.array([[yrs]])
expected_salary=regressor.predict(temp)
print(expected_salary)
