# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 22:04:08 2017

@author: alok_
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbrn

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

loan_data = pd.read_csv('loan_data.csv')

loan_data.describe()
loan_data.info() 

sbrn.heatmap(loan_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')

sbrn.pairplot(loan_data)

plt.figure(figsize=(10,6))
loan_data[loan_data['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue', bins=30,label='Credit.Policy=1')
loan_data[loan_data['credit.policy']==0]['fico'].hist(alpha=0.5,color='red', bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')

loan_data[loan_data['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue', bins=30,label='Not fully paid=1')
loan_data[loan_data['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red', bins=30,label='Not fully paid=0')
plt.legend()
plt.xlabel('FICO')

sbrn.countplot(loan_data['purpose'],hue=loan_data['not.fully.paid'])

sbrn.jointplot(loan_data['fico'],loan_data['int.rate'])


sbrn.lmplot(x='fico',y='int.rate',data=loan_data,hue='credit.policy',col='not.fully.paid',palette='Set1')


#convert categorical variable into dummy variable..
loan_data = pd.concat([loan_data,pd.get_dummies(loan_data['purpose'],drop_first=True)],axis=1)

#drop the pupose column with categorical variable...
loan_data.drop(['purpose'],axis=1,inplace=True)

#train Test Split...
loan_data_features = loan_data.drop('not.fully.paid',axis=1)
loan_paid_or_not = loan_data['not.fully.paid']
loan_data_features_train,loan_data_features_test,loan_paid_or_not_train,loan_paid_or_not_test = train_test_split(loan_data_features,loan_paid_or_not,test_size = 0.4,random_state=100)

Decision_Tree = DecisionTreeClassifier()
Decision_Tree.fit(loan_data_features_train,loan_paid_or_not_train)

loan_data_predict = Decision_Tree.predict(loan_data_features_test)

#Prinitng out the performance attributes of the algorithm...
print(confusion_matrix(loan_paid_or_not_test,loan_data_predict))
print(classification_report(loan_paid_or_not_test,loan_data_predict))

#Random Forest..
#using different way of splitting data set and fitting.. we can also use the trainig and testing set from decision tree..
#train test split...
loan_data_train,loan_data_test = train_test_split(loan_data,test_size=0.4,random_state=100)

Random_Forest = RandomForestClassifier(n_estimators=200)
Random_Forest.fit(loan_data_train.drop('not.fully.paid',axis=1),loan_data_train['not.fully.paid'])
loan_data_RF_predict = Random_Forest.predict(loan_data_test.drop('not.fully.paid',axis=1))

#Prinitng out the performance attributes of the algorithm...
print(confusion_matrix(loan_data_test['not.fully.paid'],loan_data_RF_predict))
print(classification_report(loan_data_test['not.fully.paid'],loan_data_RF_predict))

