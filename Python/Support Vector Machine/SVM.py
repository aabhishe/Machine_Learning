# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 18:49:11 2017

@author: alok_
"""

import pandas as pd
import numpy as nm
import seaborn as sbrn
import matplotlib.pyplot as pplt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

#load the Iris flower dataset data

iris_flower_ds = sbrn.load_dataset('iris')

iris_flower_ds.info()
iris_flower_ds.describe()

sbrn.pairplot(iris_flower_ds,hue='species')

#Spitting the dataset in training and testting data set....

iris_flower_train_ds,iris_flower_test_ds = train_test_split(iris_flower_ds, test_size=0.4, random_state=100)

iris_flower_SVM = SVC()
iris_flower_SVM.fit(iris_flower_train_ds.drop('species',axis=1),iris_flower_train_ds['species'])

# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
#  max_iter=-1, probability=False, random_state=None, shrinking=True,
#  tol=0.001, verbose=False)

iris_flower_predict_ds = iris_flower_SVM.predict(iris_flower_test_ds.drop('species',axis=1))

print(confusion_matrix(iris_flower_test_ds['species'],iris_flower_predict_ds))

print(classification_report(iris_flower_test_ds['species'],iris_flower_predict_ds))

# Use Grid Search to see if this result can be improved...

param_grid = {'C':[0.01,0.1,1,10,100,1000],'gamma':[1,0.1,0.001,0.0001]}

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=5)
grid.fit(iris_flower_train_ds.drop('species',axis=1),iris_flower_train_ds['species'])

iris_flower_grid_predict_ds = grid.predict(iris_flower_test_ds.drop('species',axis=1))


print(confusion_matrix(iris_flower_test_ds['species'],iris_flower_grid_predict_ds))
print(classification_report(iris_flower_test_ds['species'],iris_flower_grid_predict_ds))








