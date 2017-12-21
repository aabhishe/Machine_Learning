# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 20:51:45 2017

@author: alok_
"""

import numpy as np
import pandas as pd
import seaborn as sbrn
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

#read data - make college name as index
college_data = pd.read_csv('College_Data.csv',index_col=0)

#Explore data - checking for missing values and outliers etc..
sbrn.heatmap(college_data.isnull(),yticklabels=False,cmap='viridis',cbar=False)
college_data.head()
college_data.describe()
college_data.info()

#convert categorical variable private from yes and no to dummy variable pvt_pub with 0 for private and 1 for public..
college_data = pd.concat([college_data,pd.get_dummies(college_data['Private'])],axis=1)
college_data.rename(columns={'No':'Pvt_Pub'},inplace=True)
#delete the Yes column..
college_data = college_data.drop('Yes',axis=1)

#create a kmeans model with 2 clusters...
College_KMeans_model = KMeans(n_clusters=2)
#Fit the model exceot the categorical column of private...
College_KMeans_model.fit(college_data.drop('Private',axis=1))
#Checking the classification labels
College_KMeans_model.labels_

#Check how KMeans performed..
print(confusion_matrix(college_data['Pvt_Pub'],College_KMeans_model.labels_))
print(classification_report(college_data['Pvt_Pub'],College_KMeans_model.labels_))

