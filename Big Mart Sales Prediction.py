# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 15:29:02 2021

@author: devas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from math import sqrt
train=pd.read_csv('C:\\Users\\devas\\OneDrive\\Desktop\\Study Material\\Demo Datasets\\Lesson 4\\bigmart_train.csv')
test=pd.read_csv('C:\\Users\\devas\\OneDrive\\Desktop\\Study Material\\Demo Datasets\\Lesson 4\\bigmart_test.csv')

train['Item_Fat_Content'].unique()
train['Outlet_Establishment_Year'].unique()
train['Outlet_Age']=2018-train['Outlet_Establishment_Year']
train['Outlet_Size'].unique()

#finding frequency of each category in each nominal variable
train['Item_Fat_Content'].value_counts()
train['Outlet_Size'].mode()[0]
train['Outlet_Size']=train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0])
train['Item_Weight']=train['Item_Weight'].fillna(train['Item_Weight'].mean())
train['Item_Visibility'].hist(bins=20)
Q1=train['Item_Visibility'].quantile(0.25)
Q3=train['Item_Visibility'].quantile(0.75)
IQR=Q3-Q1
filt_train=train.query('(@Q1-1.5*@IQR)<=Item_Visibility <=(@Q3+1.5*@IQR)')

train=filt_train
#converting numerical values into categories
train['Item_Visibility_bins']=pd.cut(train['Item_Visibility'],[0.000,0.065,0.13,0.2],labels=['Low Viz','Viz','High Viz'])
train['Item_Visibility_bins'].value_counts()

#replacing null values with low visibility
train['Item_Visibility_bins']=train['Item_Visibility_bins'].replace(np.nan,'Low Viz',regex=True)

#changing representation of fat content
train['Item_Fat_Content']=train['Item_Fat_Content'].replace(['Low fat','LF'],'Low Fat')
train['Item_Fat_Content']=train['Item_Fat_Content'].replace(['Regular','reg'],'Regular')

#converting all categorical variables into numerical using label encoder
le=LabelEncoder()
train['Item_Fat_Content'].unique()
train['Item_Fat_Content']=le.fit_transform(train['Item_Fat_Content'])
train['Item_Visibility_bins']=le.fit_transform(train['Item_Visibility_bins'])
train['Outlet_Size']=le.fit_transform(train['Outlet_Size'])
train['Outlet_Location_Type']=le.fit_transform(train['Outlet_Location_Type'])

#creating dummies for outlet type
dummies=pd.get_dummies(train['Outlet_Type'])
#merging dummy and train dataframe
train=pd.concat([train,dummies],axis=1)
train.dtypes
#dropping irrelevant variables
train=train.drop(['Item_Identifier','Item_Type','Outlet_Identifier','Outlet_Type','Outlet_Establishment_Year'],axis=1)

#fitting linear model
x=train.drop('Item_Outlet_Sales',axis=1)
y=train.Item_Outlet_Sales

test['Outlet_Size']=test['Outlet_Size'].fillna('Medium')
test['Item_Visibility_bins']=pd.cut(train['Item_Visibility'],[0.000,0.065,0.13,0.2],labels=['Low Viz','Viz','High Viz'])
test['Item_Weight']=test['Item_Weight'].fillna(test['Item_Weight'].mean())
test['Item_Visibility_bins']=test['Item_Visibility_bins'].fillna('Low Viz')

test['Outlet_Size']=le.fit_transform(test['Outlet_Size'])
test['Outlet_Location_Type']=le.fit_transform(test['Outlet_Location_Type'])
test['Outlet_Age']=le.fit_transform(test['Outlet_Establishment_Year'])
dummy=pd.get_dummies(test['Outlet_Type'])
test=pd.concat([test,dummy],axis=1)

x_test=test.drop(['Item_Identifier','Item_Type','Outlet_Identifier','Outlet_Type','Outlet_Establishment_Year'],axis=1)

#fitting linear model
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=42)

lr=LinearRegression()
lr.fit(xtrain,ytrain)

lr.coef_
lr.intercept_

predictions=lr.predict(xtest)
rmse=sqrt(mean_squared_error(ytest,predictions))

from sklearn.linear_model import Ridge,Lasso,ElasticNet
ridge=Ridge(alpha=0.001)
ridge.fit(xtrain,ytrain)
pred_ridge=ridge.predict(xtest)
rmse_ridge=sqrt(mean_squared_error(ytest,predictions))
r2_ridge=r2_score(ytest,pred_ridge)

from sklearn.linear_model import Ridge,Lasso,ElasticNet
lasso=Lasso(alpha=0.001)
lasso.fit(xtrain,ytrain)
pred_lasso=lasso.predict(xtest)
rmse_lasso=sqrt(mean_squared_error(ytest,pred_lasso))
r2_lasso=r2_score(ytest,pred_lasso)


from sklearn.linear_model import Ridge,Lasso,ElasticNet
elnet=ElasticNet(alpha=0.001)
elnet.fit(xtrain,ytrain)
pred_elnet=elnet.predict(xtest)
rmse_elnet=sqrt(mean_squared_error(ytest,pred_elnet))
r2_elnet=r2_score(ytest,pred_elnet)
