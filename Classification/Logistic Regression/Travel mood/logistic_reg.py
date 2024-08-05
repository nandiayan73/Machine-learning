import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("data.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
print(x_train)
x_train[:,1:] = sc.fit_transform(x_train[:,1:])
x_test[:,1:] = sc.transform(x_test[:,1:])

print(x_train)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)


from sklearn.metrics import confusion_matrix,accuracy_score 

cm=confusion_matrix(y_test,y_pred)
# print(cm)
score=accuracy_score(y_test,y_pred)
# print(score)
