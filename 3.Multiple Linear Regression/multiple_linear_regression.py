# Multiple Linear Regression

#Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import data set
ds = pd.read_csv("50_Startups.csv")

#divide into dependent and independent variable
#first 4 column are independent variable and lat column is dependent variable

X = ds.iloc[:,:-1].values

Y = ds.iloc[:,4].values
Y = Y.reshape(-1,1) # reshape dependent variable because this will throw error while prediction

#Encoding the categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LE = LabelEncoder()
X[:,3] = LE.fit_transform(X[:,3])

#Craete dummy variable
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],
    remainder='passthrough')
X = ct.fit_transform(X)



#Avoid trap in dummy variable

X = X[:,1:]


#Splitting dataset in training and test dataset..
from sklearn.model_selection import train_test_split
train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size = 0.2, random_state = 0)







