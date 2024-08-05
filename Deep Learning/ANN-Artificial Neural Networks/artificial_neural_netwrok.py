import numpy as np
import tensorflow as tf
import pandas as pd

dataset=pd.read_csv("Churn_Modelling.csv")
x=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values

# Part-1 Data Preprocesing
# encoding the gender data:
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x[:,2]=le.fit_transform(x[:,2])

# Encoding the categorical data section:
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling is compulsory for deep learning:
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Part 2 - Building the ANN

# Initializing the ANN
ann = tf.keras.models.Sequential()

# The dense layers are the hidden layers, the units here are the no. of the hidden layers we want to make.


# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))



# Training the ann:
ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

ann.fit(x_train,y_train,batch_size=32,epochs=100)
# batch_size: compare result to prediction all together.

# Predicting the result:

y_pred=ann.predict(x_test)
y_pred=(y_pred>0.5)

print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))

# Accuracy result
from sklearn.metrics import confusion_matrix,accuracy_score
score=accuracy_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)
print(score)