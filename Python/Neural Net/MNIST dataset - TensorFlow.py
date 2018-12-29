# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 14:27:35 2017

@author: alok_
"""
#import tensorflow and matplotlib
import tensorflow as tf
import matplotlib.pyplot as plt
#import the MNIST image database for handwritten charactor recognisition...
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('C:/Users/alok_/OneDrive/Machine Learning/Python for Data Science and Machine Learning Bootcamp/Python Code',one_hot=True)

#exlplore the data...
type(mnist)
mnist.train.images.shape

#storing the 2nd image from the dataset... reshaping the matrix back to 28X28 pixel grid..
sample = mnist.train.images[2].reshape(28,28)
#displaying the image..
plt.imshow(sample,cmap='Greys')

#defining the learning parameters...
learning_rate = 0.001
training_epochs = 200
batch_size = 100
#classes of output...
n_classes = 10
#number of training set..
n_samples = mnist.train.num_examples
#number of input layer..
n_input = 784
#number of hiddent layer...
n_hidden_1=256
n_hidden_2=256


def multi_layer_perceptron(x,weights,biases):
    '''
    x=placeholder for data
    weights: dict of weights
    biases: dict of bias values
    '''
    #first hidden layer with Relu activation function...
    # y = w*x+b
    layer_1=tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    #y=relu(y)
    layer_1=tf.nn.relu(layer_1)
    
    #second hidden layer..
    layer_2=tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
    layer_2=tf.nn.relu(layer_2)
    
    #Output layer...
    out_layer = tf.matmul(layer_2,weights['out'])+biases['out']
    
    return out_layer
    

weights = {'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
            'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
            'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
            }

    
biases = {'b1':tf.Variable(tf.random_normal([n_hidden_1])),
            'b2':tf.Variable(tf.random_normal([n_hidden_2])),
            'out':tf.Variable(tf.random_normal([n_classes]))
            }

x=tf.placeholder('float',[None,n_input])
y=tf.placeholder('float',[None,n_classes])
pred=multi_layer_perceptron(x,weights,biases)

cost= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

t = mnist.train.next_batch(10)

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(training_epochs):
    avg_cost = 0.0
    total_batch=int(n_samples/batch_size)
    for i in range(total_batch):
        batch_x,batch_y=mnist.train.next_batch(batch_size)
        _,c=sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})
        avg_cost+=c/total_batch
    #print("Epoch:{} cost{:.4f}".format(epoch+1,avg_cost))
print("Model has completed {} epcochs of training".format(training_epochs))


correct_predictions=tf.equal(tf.argmax(pred,1),tf.arg_max(y,1))
correct_predictions=tf.cast(correct_predictions,'float')
accuracy=tf.reduce_mean(correct_predictions)

accuracy.eval({x:mnist.test.images,y:mnist.test.labels})
