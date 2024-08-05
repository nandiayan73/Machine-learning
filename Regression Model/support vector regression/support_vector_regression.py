import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# importing the dataset
dataset=pd.read_csv("Position_Salaries.csv")
x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values


y = y.reshape(len(y),1)#Changing y into a 2-D matrix.
print(y)


# Feature Scaling
    # 1.Apply only after the split,
    # 2.aplly only on the large data like salary here,
    # 3.Don't need to apply on the binary values, resulted from one hunt coding.  
from sklearn.preprocessing import StandardScaler

    # Standardizaton gives value from -3 to +3

sc_x=StandardScaler()
sc_y=StandardScaler()

x=sc_x.fit_transform(x)#fit_transform is used only once as the fit fits only once
y=sc_y.fit_transform(y)


#Trainng the support vector regressor model:
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(x,y)


# Training the polynomial regressor model:

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(dataset.iloc[:,1:-1].values)

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x_poly,dataset.iloc[:,-1].values)


# Predicting a new result:
# as the fit was made with a scaled input and output, the result of the predict fn. would also be a scaled value
# we will have to unscale the value to get the required output.

# 1.Predicting the svr result
scaled_res=regressor.predict(sc_x.transform([[6.5]]))
res=sc_y.inverse_transform(scaled_res.reshape(-1,1))

# 2.Predicting the polynomial regression result

res2=lin_reg.predict(poly_reg.transform([[6.5]]))
print("The salary of level:"+str(6.5)+" "+str(res))
print("The salary of level:"+str(6.5)+" "+str(res2))

# visualising the svr result

# 1.Scaled value visualization

plt.scatter(x,y,color="red")
plt.plot(x,regressor.predict(x),color="blue")
plt.title("SVR model visualization(scaled)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

# 2.Visualization of unscaled value

plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y),color="red")
plt.plot(sc_x.inverse_transform(x),sc_y.inverse_transform(regressor.predict(x).reshape(-1,1)),color="black")
plt.title("SVR model visualization(unscaled)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

# Visualising the polynomial regression result

plt.scatter(dataset.iloc[:,1:-1].values,dataset.iloc[:,-1],color="red")
plt.plot(dataset.iloc[:,1:-1].values,lin_reg.predict(x_poly),color="blue")
plt.title("Polynomial model visualization")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()


