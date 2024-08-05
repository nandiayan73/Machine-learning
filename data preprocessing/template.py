# dataset preprocessing

# importing libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the datasets


dataset=pd.read_csv("D:\CODING\AIML\learnings\data preprocessing\Data.csv")
x=dataset.iloc[:,:-1].values
# x=> the features of the dataset
# dataset.iloc[row,column]
# here selecting all the rows => :
# selecting all the clomns except the last one => :-1

y=dataset.iloc[:,-1].values
# print(y)
# y=> the depend variable that depends on the feature
print(x)
# [['France' 44.0 72000.0]
#  ['Spain' 27.0 48000.0]
#  ['Germany' 30.0 54000.0]
#  ['Spain' 38.0 61000.0]
#  ['Germany' 40.0 nan]
#  ['France' 35.0 58000.0]
#  ['Spain' nan 52000.0]
#  ['France' 48.0 79000.0]
#  ['Germany' 50.0 83000.0]
#  ['France' 37.0 67000.0]]