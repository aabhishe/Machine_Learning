# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 19:18:40 2017

@author: alok_
"""
#import libararies
import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#load the data
ecommerce_data = pd.read_csv('Ecommerce Customers.csv')

#data explorations
ecommerce_data.info()
ecommerce_data.describe()
ecommerce_data.head()
ecommerce_data.columns

#data visulaization:
sb.jointplot(ecommerce_data['Time on Website'],ecommerce_data['Yearly Amount Spent'])
sb.jointplot(ecommerce_data['Time on App'],ecommerce_data['Yearly Amount Spent'])
sb.jointplot(x='Time on App',y='Length of Membership',data=ecommerce_data,kind='hex')
sb.pairplot(ecommerce_data)
sb.heatmap(ecommerce_data.corr(),annot=True)
sb.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=ecommerce_data)

#Create the predication and predicators variables data frame
ecommerce_data_predictors = ecommerce_data[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
ecommerce_data_annual_amt_spent = ecommerce_data['Yearly Amount Spent']

#Split the data in testing and training set
ecommerce_data_predictors_train, ecommerce_data_predictors_test, ecommerce_data_annual_amt_spent_train, ecommerce_data_annual_amt_spent_test = train_test_split(ecommerce_data_predictors, ecommerce_data_annual_amt_spent,test_size = 0.4,random_state=101)

#Linear Regression Model
linear_model_ecommerce_data= LinearRegression()
#Trainig the linear regression model
linear_model_ecommerce_data.fit(ecommerce_data_predictors_train,ecommerce_data_annual_amt_spent_train)
#print the coefficients
print(linear_model_ecommerce_data.coef_)
print(linear_model_ecommerce_data.intercept_)

coeff_df = pd.DataFrame(linear_model_ecommerce_data.coef_,ecommerce_data_predictors.columns,columns=['coefficent'])
coeff_df
#Testing the linear regression model - prediction on testing data set
linear_model_ecommerce_prediction = linear_model_ecommerce_data.predict(ecommerce_data_predictors_test)
#Plot the predicted value Vs the actual value
plt.scatter(ecommerce_data_annual_amt_spent_test,linear_model_ecommerce_prediction)

sb.distplot((ecommerce_data_annual_amt_spent_test-linear_model_ecommerce_prediction),bins=30)

#Linear Model performance analysis
np.sqrt(metrics.mean_squared_error(ecommerce_data_annual_amt_spent_test,linear_model_ecommerce_prediction))

metrics.mean_absolute_error(ecommerce_data_annual_amt_spent_test,linear_model_ecommerce_prediction)

metrics.mean_squared_error(ecommerce_data_annual_amt_spent_test,linear_model_ecommerce_prediction)




