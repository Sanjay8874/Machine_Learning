#data preprocessing

#import libraries
import numpy as np # use to mathemetical
import pandas as pd # use to dataSet
import matplotlib.pyplot as plt # use to plot gragph

#import dataset
dataset = pd.read_csv("Data.csv")

#Two variable dependent(Y) and independent(X)
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values


#Taking care of missing dataset
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.NaN, strategy = "mean")
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:,1:3])


#Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X= LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
#onehotencoder = OneHotEncoder(categories = [0])
#X = onehotencoder.fit_transform(X).toarray() # In the new version this line not execute properly.
#X = pd.get_dummies(X[:,0])

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],
    remainder='passthrough')
X = ct.fit_transform(X)

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


#Splitting dataset in training and test dataset..
from sklearn.model_selection import train_test_split
train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size = 0.2, random_state = 0)


#Features Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
train_X = sc_X.fit_transform(train_X)
test_X = sc_X.transform(test_X)


























