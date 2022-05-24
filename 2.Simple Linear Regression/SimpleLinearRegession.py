#Simple Linear regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Import dataset
dataset = pd.read_csv("Salary_Data.csv")

#Divide data in dependent and not dependent variable ie. y = mx + c

X = dataset.iloc[:,0].values
Y = dataset.iloc[:,1].values


#Splitting dataset in training and test dataset..
from sklearn.model_selection import train_test_split
train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size = 1/3, random_state = 0)

#Scatter plot dataset
plt.scatter(X,Y, c = "red", marker = "*")
plt.title("Scatter plot")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

'''#Scatter plot for tarin and test dataset
plt.scatter(test_X,test_Y, c = "red", marker = "*")
plt.title("Scatter plot")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()'''

#Check dimention of array
#train_X.ndim
#train_Y.ndim

train_X = train_X.reshape(-1,1)# getting error "Expected 2D array, got 1D array instead"
test_X = test_X.reshape(-1,1)


#Fitting simple linearb regression to training dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(train_X, train_Y)

#predicting Test data

Y_pred = regressor.predict(test_X)
train_pred = regressor.predict(train_X)


#visualisation of tarining data

plt.scatter(train_X,train_Y,c = "red",marker="*")
plt.plot(train_X,regressor.predict(train_X),c = "blue")
plt.title("Salary prediction")
plt.xlabel("Experience")
plt.ylabel("salary")
plt.show()


#visualisation of testing data

plt.scatter(test_X,test_Y,c = "red",marker="*")
plt.plot(train_X,regressor.predict(train_X),c = "blue")
plt.title("Salary prediction")
plt.xlabel("Experience")
plt.ylabel("salary")
plt.show()


















