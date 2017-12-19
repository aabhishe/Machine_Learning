# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 19:28:46 2017

@author: alok_
"""
#import Libraries
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib as mtl
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#read the data
advertising_data = pd.read_csv('advertising.csv')

#Data Exploration
advertising_data.head()
advertising_data.info()
advertising_data.describe()

#create a heatmap of data to check if any data is missing or null
sn.heatmap(advertising_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#More data exploration and check on different predictor and their correlation
sn.distplot(advertising_data['Age'],bins=30)
sn.jointplot(advertising_data['Age'],advertising_data['Area Income'])
sn.kdeplot(advertising_data['Age'],advertising_data['Daily Time Spent on Site'],shade='red')
sn.jointplot(advertising_data['Age'],advertising_data['Daily Time Spent on Site'],kind='kde',color='red')
sn.pairplot(advertising_data,hue='Clicked on Ad',palette='bwr')
sn.heatmap(advertising_data.corr(),annot=True)

#split the data into training set and testing set...
advertising_data_train,advertising_data_test = train_test_split(advertising_data,test_size = 0.4,random_state=101)

#Logistic Regression model
logistic_model = LogisticRegression()
#logistic regression model fit/training
logistic_model.fit(advertising_data_train[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']],advertising_data_train['Clicked on Ad'])
#Logistic regression mdoel prediction
advertising_data_prediction = logistic_model.predict(advertising_data_test[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']])
#Prediction accuracy report
print(classification_report(advertising_data_test['Clicked on Ad'],advertising_data_prediction))
#Model prediction performance confusion matrix
confusion_matrix(advertising_data_test['Clicked on Ad'],advertising_data_prediction)
#Model co-efficient..
print(logistic_model.coef_)
print(logistic_model.intercept_)


