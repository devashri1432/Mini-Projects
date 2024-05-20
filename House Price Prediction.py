# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 16:08:14 2021

@author: devas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
data=pd.read_csv('C:\\Users\\devas\\OneDrive\\Desktop\\Start Tech Academy\\Data Files\\1. ST Academy - Crash course and Regression files\\House_Price.csv')
data.head(10)
data.shape
edd=data.describe()
edd
"""
sb.jointplot(x='n_hot_rooms',y='price',data=data)
sb.jointplot(x='rainfall',y='price',data=data)
data.head()
#sb.countplot(x='airport',data=data)
#sb.countplot(x='waterbody',data=data)
sb.countplot(x='bus_ter',data=data)
"""
"""
1) Missing values
2)Skewness or outliers in crime rate
3)Outliers in hotel rooms and rainfall
4) bus terminal has only 1 value
"""
data.info()
#outlier treatment and identification
ans=np.percentile(data.n_hot_rooms,[99])
ans
up=np.percentile(data.n_hot_rooms,[99])[0]
d1=data[(data.n_hot_rooms>up)]
d1=data.n_hot_rooms[(data.n_hot_rooms>3*up)]=3*up

lv=np.percentile(data.rainfall,[1])[0]
data[(data.rainfall<lv)]
data.rainfall[(data.rainfall<0.3*lv)]=0.3*lv
data.rainfall

sb.jointplot(x='crime_rate',y='price',data=data)
data.describe()

"""
Missing values implementation
"""
data.info()
data.n_hos_beds=data.n_hos_beds.fillna(data.n_hos_beds.mean())
data.info()

#data=data.fillna.(data.mean())

#variable transformation and deletion in python
sb.jointplot(x='crime_rate',y='price',data=data)
data.crime_rate=np.log(data.crime_rate+1)
sb.jointplot(x='crime_rate',y='price',data=data)
data['avg_dist']=(data.dist1+data.dist2+data.dist3+data.dist4)/4
data.describe()

del data['dist1']
del data['dist2']
del data['dist3']
del data['dist4']
data.describe()

del data['bus_ter']
data.head(5)

#creating dummy variables 
data=pd.get_dummies(data)

#deleting waterbody_none and airport_NO as they are redundant variables
del data['waterbody_None']
del data['airport_NO']

#analyzing correlation matrix
cm=data.corr()
del data['parks']

#fitting liner model
#Method-1
"""
import statsmodels.api as sn
x=sn.add_constant(data['room_num'])
lm=sn.OLS(data['price'],x).fit()
lm.summary()
"""
#Method-2 (most commonly used method)
from sklearn.linear_model import LinearRegression
import numpy as np
x=data[['room_num']]
y=data['price']
lm2=LinearRegression()
lm2.fit(x,y)
print(lm2.intercept_,lm2.coef_)
#help(lm2)

y_pred=lm2.predict(x)

#plotting the linear regression model
#help(sb.jointplot)
sb.jointplot(x=data['room_num'],y=data['price'],data=data,kind='reg')

#fitting multiple linear regression model
#method 1
#import statsmodels.api as sn
x_multi=data.drop('price',axis=1)
y_multi=data['price']
#x_multi_cons=sn.add_constant(x_multi)
"""
lm=sn.OLS(y_multi,x_multi_cons).fit()
lm.summary()
"""
#method2
lm3=LinearRegression()
lm3.fit(x_multi,y_multi)
print(lm3.intercept_,lm3.coef_)

#splitting the datset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_multi,y_multi,test_size=0.2,random_state=0)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

#fitting linear model
lm_a=LinearRegression()
lm_a.fit(x_train,y_train)

#predicting the values
y_test_a=lm_a.predict(x_test)
y_train_a=lm_a.predict(x_train)

#accuracy of the model and evaluating the performance of our model
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_test_a)
r2_train=r2_score(y_train,y_train_a)

#ridge and lasso regression
from sklearn import preprocessing
#standardizing the variables
sc=preprocessing.StandardScaler().fit(x_train)
x_train_s=sc.transform(x_train)
x_test_s=sc.transform(x_test)

from sklearn.linear_model import Ridge
lm_r=Ridge(alpha=0.5) #alpha is the same as lambda
lm_r.fit(x_train_s,y_train)

r2_score(y_test,lm_r.predict(x_test_s))

#to find out various values of lambda and their corresponding r2 score
from sklearn.model_selection import validation_curve
#to find the optimum value of alpha for the best fit we use the validation curve to find out various lambda values and find the maximum from it and fit the model for that value
param_range=np.logspace(-2,8,100)
param_range
train_scores,test_scores=validation_curve(Ridge(), x_train_s, y_train, param_name="alpha", param_range=param_range,scoring='r2')
print(train_scores)
train_mean=np.mean(train_scores,axis=1)
test_mean=np.mean(test_scores,axis=1)
max(test_mean)
sb.jointplot(x=np.log(param_range),y=test_mean)

#to find the location of maximum r2 value
np.where(test_mean==max(test_mean))
param_range[31]

#fitting the best ridge model with this lambda value
lm_r_best=Ridge(alpha=param_range[31])
lm_r_best.fit(x_train_s,y_train)

r2_score(y_test,lm_r_best.predict(x_test_s))
r2_score(y_train,lm_r_best.predict(x_train_s))
