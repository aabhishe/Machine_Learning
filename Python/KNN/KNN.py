# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 00:07:33 2017

@author: alok_
"""
#import libraries
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
#read the data
project_data = pd.read_csv('KNN_Project_Data.csv')
#Explore the data
project_data.head()
project_data.info()
project_data.describe()
#Check for missing data
sb.heatmap(project_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#normalize features... excluding the prediction
normalized = StandardScaler()
normalized.fit(project_data.drop('TARGET CLASS',axis=1))
feature_normalized = normalized.transform(project_data.drop('TARGET CLASS',axis=1))
df_normalized_features = pd.DataFrame(feature_normalized,columns=project_data.columns[:-1])
#Stictch the data frame together after normalization
df_normalized_features_merged = pd.concat([df_normalized_features,project_data['TARGET CLASS']],axis=1)

#Split the data in testing set and training set..
project_data_train,project_data_test = train_test_split(df_normalized_features_merged,test_size = 0.4,random_state=100)
#Use elbow method to find the best n value for low error rate.
error_rate = []
for n in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(project_data_train[['XVPM', 'GWYH', 'TRAT', 'TLLZ', 'IGGA', 'HYKR', 'EDFS', 'GUUB', 'MGJM','JHZC', 'TARGET CLASS']],project_data_train['TARGET CLASS'])
    project_data_predict = knn.predict(project_data_test[['XVPM', 'GWYH', 'TRAT', 'TLLZ', 'IGGA', 'HYKR', 'EDFS', 'GUUB', 'MGJM','JHZC', 'TARGET CLASS']])
    error_rate.append(np.mean(project_data_predict!=project_data_test['TARGET CLASS']))

#Elbow Plot for visualization of n value Vs error
plt.figure(figsize=(10,6))
plt.plot(range(1,50),error_rate,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

#Setting up KNN with n =10
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(project_data_train[['XVPM', 'GWYH', 'TRAT', 'TLLZ', 'IGGA', 'HYKR', 'EDFS', 'GUUB', 'MGJM','JHZC', 'TARGET CLASS']],project_data_train['TARGET CLASS'])
project_data_predict = knn.predict(project_data_test[['XVPM', 'GWYH', 'TRAT', 'TLLZ', 'IGGA', 'HYKR', 'EDFS', 'GUUB', 'MGJM','JHZC', 'TARGET CLASS']])

#Prinitng out the performance attributes of the algorithm...
print(confusion_matrix(project_data_test['TARGET CLASS'],project_data_predict))
print(classification_report(project_data_test['TARGET CLASS'],project_data_predict))



