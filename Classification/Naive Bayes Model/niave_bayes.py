import pandas as pd
import numpy as np

dataset=pd.read_csv("data.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)



from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix
matrix=confusion_matrix(y_test,y_pred)
print(matrix)
score=accuracy_score(y_test,y_pred)
print(score)

