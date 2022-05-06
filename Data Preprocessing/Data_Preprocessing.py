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