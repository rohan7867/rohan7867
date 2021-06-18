## Importing required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

url="http://bit.ly/w-data"
data=pd.read_csv(url)
data

## Scatter plot of the dataset

%matplotlib inline
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.scatter(data.Hours,data.Scores,color="black")

x=data[['Hours']] #independent variable
y=data[["Scores"]] #dependent variable

## Splitting the dataset into train set and test set

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

lr=LinearRegression()

lr.fit(x_train,y_train)

## Actual values and predicted values

y_pred=lr.predict(x_test)

y_test #actual values

y_pred  # predicted values

from sklearn.metrics import mean_squared_error

mean_squared_error(y_test,y_pred)

## Fitting of the simple regression model

m=lr.coef_ #regression coefficient
b=lr.intercept_  #regression intercept
line=m*(x_train)+b
plt.scatter(x_train,y_train)
plt.plot(x_train,line)
plt.show()

## Prediction

pr=lr.predict(y_test)
list(zip(y_test,pr))
hour=[9.25]
prediction=lr.predict([hour])
print("no of hour={}",format([hour]))
print("predicted score={}",format(prediction[0]))

# Result :

## If a student studies for 9.25 hrs/day then he/she get 93.63% scores

