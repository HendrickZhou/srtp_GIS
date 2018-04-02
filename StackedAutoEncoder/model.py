# -*- coding: utf-8 -*-
"""
This file contains the model of stacked auto encoder
@author Zhou Hang
"""
import functools
import copy
import tensorflow as tf
import numpy as np
import data as dt

# define the property function to make the code cleaner
def lazy_property(function):
    attribute = '_cache_' + function.__name__
    
    @property
    @functools.wrap(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name__):
                setattr(self, attribute, function(self))
        return decorator
        
# kits
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)
def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

# define the model
class model(object):
    def __init__(self, data_set):
        '''
        @input:    instance data_set
        @output:   none
        '''
        # receive the data set
        self.data_set = data_set
        self.input = data_set.data
        self.label = data_set.label
        self.class_num = data_set.class_num 
        
        # store the params of the nn
        self.W = []
        self.b = []
        
        # interface
        self.predict
        self.train
        #self.fineTuning
        #self.error
    
    #construct the nerual network
    @property
    def train(self):
        '''
        babala
        '''
        # initialize the hyperparam
        data_set_num = len(self.input)
        batch_size = 512
        data_length = len[input[0]]       # this can be automated in the later version
        iteration = data_set_num / batch_size
        
        h1_num = 100 
        h2_num = 50
        learning_rate1 = 0.01
        learning_rate2 = 0.01
        learning_rate3 = 0.01
        
        # initalize the param
        W1 = weight_variable([data_length, h1_num])
        b1 = bias_variable([h1_num])
        W1_ = weight_variable([h1_num, data_length])
        b1_ = bias_variable([data_length])
        
        W2 = weight_variable([h1_num, h2_num])
        b2 = bias_variable([h2_num])
        W2_ = weight_variable([h2_num, h1_num])
        b2_ = bias_variable([h1_num])
        
        W_out = weight_variable([h2_num, self.class_num])
        b_out = bias_variable([self.class_num])
        
        # construct the graph of the first encoder
        X1 = tf.placeholder("float", [None, data_length])# none means the number is unsure
        h1_out = tf.nn.sigmod(tf.matmul(X1, W1) + b1)
        X1_ = tf.nn.sigmod(tf.matmul(h1_out, W1_) + b1_)
        
        cost1 =  tf.reduce_sum(tf.pow(X1_-X1 , tf.to_float(tf.convert_to_tensor\
        (2 * np.ones([data_length]))))) / batch_size
        train_1 = tf.train.GradientDescentOptimizer(learning_rate1).minimize(cost1)
        sess = tf.Session()
        sess.run(tf.initialize_variables(W1, b1, W1_, b1_))
        
        # train the fisrt encoder
        for i in xrange(iteration):
            batch_x,batch_y = data_set.next_batch(batch_size) # does this mean: the batch_x will still be conveyed to nn one by one?
            sess.run(train1, feed_dict = {X1 : batch_x})
        
        # prepare for the next layer
        tf.to_float(np.reshape(self.input, [data_set_num, data_length])) # use reshape we can transfer the list to array
        encoder1_out = tf.nn.sigmod(tf.matmul(self.input, W1) + b1)
        input2 = encoder1_out
        
        # construct the graph of the second encoder
        X2 = tf.placeholder("float", [None, h1_num])
        h2_out = tf.nn.sigmod(tf.matmul(X2, W2) + b2)
        X2_ = tf.nn.sigmod(tf.matmul(h2_out, W2_) + b2_)
        
        cost2 = tf.reduce_sum(tf.pow(X2_-X2, tf.to_float(tf.convert_to_tensor\
        (2 * np.ones([h1_num]))))) / batch_size
        train_2 = tf.train.GradientDescentOptimizer(learning_rate2).minimize(2)
        sess.run(tf.initialize_variable(W2, b2, W2_, b2_))
        
        # train the second encoder
        for i in xrange(iteration):
            batch_x,batch_y = dt.next_batch(input2, batch_size) # does this mean: the batch_x will still be conveyed to nn one by one?
            sess.run(train2, feed_dict = {X2 : batch_x})
        
        # prepare for the output layer
        encoder2_out = tf.nn.sigmod(tf.matmul(input2, W2) + b2)
        input3 = encoder2_out

        # construct the output layer        
        X_out = tf.placeholder("float", shape = [None, h2_num])
        Y_out = tf.nn.softmax(tf.matmul(x_out, W_out) + b_out)
        Y_label = tf.placeholder("float", shape = [None, self.class_num])
        cost_out = tf.reduce_sum(Y_label * tf.log(Y_out)) / batch_size
        train_out = tf.train.GradientDescentOptimizer.minimize(learning_rate3)
        sess.run(tf.initialize_variable(W_out, b_out))
        
        # evaluate
        correction_prediction = tf.equal(tf.argmax(Y_out, 1), tf.argmax(Y_label, 1))
        accuracy = tf.reduce_mean(tf.cast(correction_prediction, "float"))
        
        # train and show the evaluate dynamicly
        for i in xrange(iteration):
            batch_X, batch_Y = next_batch(input3, self.label, batch_size)
            sess.run(train_out, feed_dict = {Y_label : batch_Y, X_out : batch_X})
            print("the accuracy of the No.%d iteration is %d" %(iteration, accuracy.eval(feed_dict = {Y_out : batch_Y, Y_label : batch_X}))
        
        # store the param
        W_all = [W1, W1_, W2, W2_, W_out]
        b_all = [b1, b1_, b2, b2_, b_out]
        self.W = copy.deepco
    
    @property
    def predict(self):
        