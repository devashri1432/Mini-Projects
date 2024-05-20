# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 09:20:56 2021

@author: devas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
df=pd.read_csv('C:\\Users\\devas\\OneDrive\\Desktop\\Start Tech Academy\\Data Files\\3. ST Academy - Decision Trees resource files\\Movie_classification.csv')
df.info()

#missing values implementation
df.Time_taken=df.Time_taken.fillna(df.Time_taken.mean())
df.info()

#Dummy variable creation
df=pd.get_dummies(df,columns=['3D_available','Genre'],drop_first=True)

#x-y split
x=df.loc[:,df.columns!='Start_Tech_Oscar']
y=df['Start_Tech_Oscar']

#train-test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2,random_state=0)

#fitting classification tree
from sklearn.tree import DecisionTreeClassifier
classtree=DecisionTreeClassifier(max_depth=3)
classtree.fit(x_train, y_train)

#predicting the values 
y_train_pred=classtree.predict(x_train)
y_test_pred=classtree.predict(x_test)

#model evaluation and accuracy
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix( y_test, y_test_pred)
acc=accuracy_score( y_test, y_test_pred)

"""
#plotting the classification tree
from sklearn import tree
dot_data=tree.export_graphviz(classtree, out_file=None,feature_names=x_train.columns,filled=True)#filled=True helps in conditional formatting
from IPython.display import Image
import pydotplus as py
graph=py.graph_from_dot_data(dot_data)
Image(graph.create_png()) #blue color in image represents high purity of ones and orange represents high purity on zeros and white represents no purity


Controlling tree growth
from sklearn.tree import DecisionTreeClassifier
classtree2=DecisionTreeClassifier(max_depth=4,min_samples_leaf=20)
classtree2.fit(x_train, y_train)

from sklearn import tree
dot_data=tree.export_graphviz(classtree2, out_file=None,feature_names=x_train.columns,filled=True)#filled=True helps in conditional formatting
from IPython.display import Image
import pydotplus as py
graph2=py.graph_from_dot_data(dot_data)
Image(graph2.create_png())
y_pred2=classtree2.predict(x_test)
acc2=accuracy_score( y_test, y_pred2)
"""
#1)Bagging
"""
In bagging we create a full tree
"""
from sklearn import tree
clftree=tree.DecisionTreeClassifier()
from sklearn.ensemble import BaggingClassifier
bag_clf=BaggingClassifier(base_estimator=clftree,n_estimators=1000,bootstrap=(True),n_jobs=-1,random_state=42)
bag_clf.fit(x_train,y_train)
y_bag_pred=bag_clf.predict(x_test)
cm_bag=confusion_matrix( y_test, y_bag_pred)
acc_bag=accuracy_score( y_test, y_bag_pred)

#2)Random Forest
from sklearn import tree
random=tree.DecisionTreeClassifier()
from sklearn.ensemble import RandomForestClassifier
ran_clf=RandomForestClassifier(n_estimators=1000,n_jobs=-1,random_state=42)
ran_clf.fit(x_train,y_train)
y_ran_pred=ran_clf.predict(x_test)
cm_ran=confusion_matrix( y_test, y_ran_pred)
acc_ran=accuracy_score( y_test, y_ran_pred)

#optimizing the hyper parameters
from sklearn.model_selection import GridSearchCV
rf_clf=RandomForestClassifier(n_estimators=250,random_state=42)
par={'max_features':[4,5,6,7,8,9,10],'min_samples_split':[2,3,10]}
gsv=GridSearchCV(rf_clf,par,n_jobs=-1,cv=5,scoring='accuracy')
gsv.fit(x_train,y_train)
gsv.best_params_
best=gsv.best_estimator_
y_best=best.predict(x_test)
cm_best=confusion_matrix( y_test, y_best)
acc_best=accuracy_score( y_test, y_best)

#3)Boosting

#a)Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
gbc.fit(x_train,y_train)
y_gbc_pred=gbc.predict(x_test)
acc_gbc=accuracy_score( y_test, y_gbc_pred)

from sklearn.ensemble import GradientBoostingClassifier
gbc1=GradientBoostingClassifier(learning_rate=(0.02),n_estimators=(1000),max_depth=1)
gbc1.fit(x_train,y_train)
y_gbc_pred2=gbc1.predict(x_test)
acc_gbc2=accuracy_score( y_test, y_gbc_pred2)

#optimizing hyper parameters
from sklearn.model_selection import GridSearchCV
par1={'learning_rate':[0.01,0.1,0.001],'n_estimators':[500,750,1000],'max_depth':[1,2,3,4,5]}
gsv1=GridSearchCV( estimator=gbc, param_grid=par1,cv=5,n_jobs=-1,scoring='accuracy')
gsv1.fit(x_train,y_train)
best_params=gsv1.best_params_
best_model=gsv1.best_estimator_
y_best1=best_model.predict(x_test)
cm_best1=confusion_matrix( y_test, y_best1)
acc_best1=accuracy_score( y_test, y_best1)
"""
#b)Ada Boosting
"""
from sklearn.ensemble import AdaBoostClassifier
abc=AdaBoostClassifier()
abc.fit(x_train,y_train)
y_abc_pred=abc.predict(x_test)
acc_abc=accuracy_score( y_test, y_abc_pred)

from sklearn.ensemble import AdaBoostClassifier
abc1=AdaBoostClassifier(learning_rate=0.05,n_estimators=1000)
abc1.fit(x_train,y_train)
y_abc_pred2=abc1.predict(x_test)
acc_abc2=accuracy_score( y_test, y_abc_pred2)

#optimizing hyper parameters
from sklearn.model_selection import GridSearchCV
par2={'learning_rate':[0.01,0.1,0.001],'n_estimators':[500,750,1000],'max_depth':[1,2,3,4,5]}
gsv2=GridSearchCV( estimator=abc1, param_grid=par2,cv=5,n_jobs=-1,scoring='accuracy')
gsv2.fit(x_train,y_train)
best_params1=gsv2.best_params_
best_model1=gsv2.best_estimator_
y_best2=best_model1.predict(x_test)
cm_best2=confusion_matrix( y_test, y_best2)
acc_best2=accuracy_score( y_test, y_best2)

import xgboost as xgb

xgb_clf=xgb.XGBClassifier(max_depth=5,n_estimators=10000,learning_rate=0.3,n_jobs=-1)
xgb_clf.fit(x_train,y_train)
y_pred=xgb_clf.predict(x_test)

from sklearn.metrics import accuracy_score
acc=accuracy_score( y_test, y_pred)

"""
#this method gives importance of relative variables that we are using
xgb.plot_importance(xgb_clf)
"""

#optiming xgb classifier with the hyper parameters
xc=xgb.XGBClassifier()
par={'max_depth':range(3,10,2),'gamma':[0.1,0.2,0.3],'subsample':[0.8,0.9],'colsample_bytree':[0.8,0.9],'reg_alpha':[1e-2,0.1,1]}
#subsample-subset of dataset
#colsample_bytree=instead of data we use 805 of features to create each tree
#reg_alpha=regularization parameter

from sklearn.model_selection import GridSearchCV
gsv=GridSearchCV(estimator=xc, param_grid=par,n_jobs=-1,scoring='accuracy',cv=5)
gsv.fit(x_train,y_train)
best_params=gsv.best_params_
best_model=gsv.best_estimator_
y_pred_gsv=gsv.predict(x_test)
acc_best=accuracy_score( y_test, y_pred_gsv)

