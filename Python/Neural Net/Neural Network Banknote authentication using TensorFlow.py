# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 15:35:20 2017

@author: alok_
"""
#import libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.contrib.learn.python.learn.datasets import mnist 


#load the dataset
banknotedata = pd.read_csv('data_banknote_authentication.csv')
#create training and testing dataset...
banknotedata_train,banknotedata_test = train_test_split(banknotedata,test_size=0.3,random_state=100)
#explore the data...
banknotedata.shape
banknotedata.info()
banknotedata.describe()
banknotedata_train.shape
banknotedata_train.info()
banknotedata_train.describe()

#define the neural network learning parameters:
learning_rate = 0.01                                                         #learning rate
training_epochs = 200                                                        #number of iterations
batch_size = 60                                                              #batch size for training
n_classes=2                                                                  #number of different types of outputs...
n_samples = banknotedata_train.shape[0]                                      #number of training set...
n_input = banknotedata.shape[1]                                              #number of input layer - features for each input
n_hidden_1=256                                                               #number of hidden layer#1..
n_hidden_2=128                                                               #number of hidden layer#2..
n_hidden_3=64                                                                #number of hidden layer#3..


#define weight and bias...
weights = {'w1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
           'w2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
           'w3':tf.Variable(tf.random_normal([n_hidden_2,n_hidden_3])),
           'out':tf.Variable(tf.random_normal([n_hidden_3,n_classes]))
           }

bias = {'b1':tf.Variable(tf.random_normal([n_hidden_1])),
        'b2':tf.Variable(tf.random_normal([n_hidden_2])),
        'b3':tf.Variable(tf.random_normal([n_hidden_3])),
        'out':tf.Variable(tf.random_normal([n_classes]))
        }

x=tf.placeholder('float',[None,n_input])
y=tf.placeholder('float',[None,n_classes])

def multi_layer_perceptron(x,weights,bias):
    layer_1 = tf.add(tf.matmul(x,weights['w1']),bias['b1'])
    layer_1 = tf.nn.relu(layer_1)
    
    layer_2=tf.add(tf.matmul(layer_1,weights['w2']),bias['b2'])
    layer_2=tf.nn.relu(layer_2)
    
    layer_3=tf.add(tf.matmul(layer_2,weights['w3']),bias['b3'])
    layer_3=tf.nn.relu(layer_3)
    
    out_layer=tf.add(tf.matmul(layer_3,weights['out']),bias['out'])
    
    return out_layer

pred = multi_layer_perceptron(x,weights,bias)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)    

for epoch in range(training_epochs):
    avg_cost = 0.0
    batch_x=banknotedata_train[['variance','skewness','curtosis','entropy','class']]
    batch_y=banknotedata_train[['class']]
    _,c=sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})
    avg_cost+=c
    print("Epoch:{} cost{:.4f}".format(epoch+1,avg_cost))
print("Model has completed {} epcochs of training".format(training_epochs))


correct_predictions=tf.equal(tf.argmax(pred,1),tf.arg_max(y,1))
correct_predictions=tf.cast(correct_predictions,'float')
accuracy=tf.reduce_mean(correct_predictions)

accuracy.eval({x:banknotedata_train,y:banknotedata_test})


