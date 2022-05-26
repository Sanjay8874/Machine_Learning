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


#Fitting Multiple Linear Regression to training dataset
from sklearn.linear_model import LinearRegression
LR = LinearRegression()#LR is object of class LinearRegression
LR.fit(train_X,train_Y)

#Predicting test set result
Y_Prediction = LR.predict(test_X)

#Building optimal model using backward elimination
#sometime all independent variable are not usefull to prediction. so we can eliminate remove those column which not usefull.
import statsmodels.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X,axis = 1)
#X_opt = X[:,[0,1,2,3,4,5]] #not working use bellow line 
X_opt = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
#print(regressor_OLS.summary())

# x2>0.05 so we need to remove x2 column(As per old X Table value)
X_opt = np.array(X[:, [0, 1,3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
#print(regressor_OLS.summary())

# x1>0.05 so we need to remove x1 column(As per old X Table value)
X_opt = np.array(X[:, [0,3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
#print(regressor_OLS.summary())

# x4>0.05 so we need to remove x4 column(As per old X Table value)
X_opt = np.array(X[:, [0,3, 5]], dtype=float)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
#print(regressor_OLS.summary())

# x5>0.05 so we need to remove x5 column(As per old X Table value)
X_opt = np.array(X[:, [0,3]], dtype=float)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
print(regressor_OLS.summary())









