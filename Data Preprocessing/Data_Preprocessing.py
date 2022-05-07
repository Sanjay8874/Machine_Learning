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

