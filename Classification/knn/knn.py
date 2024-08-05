import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset=pd.read_csv("knn/data.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# Feature scaling:
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


print(x_train)
# Making the model:

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric="minkowski",p=2)
# n_neighbors= total no. of neighbors.
# minkowski= this calculates the euclidean distance.
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

print(classifier.predict(sc.transform([[30,87000]])))

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test,y_pred))



