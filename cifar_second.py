#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 09:40:34 2018

@author: andre
"""

####################################
#SECOND NEURAL NETWORK FOR CIFAR-10#
####################################

import numpy as np
import tensorflow as tf


#####################
#OPENING THE DATASET#
#####################

#function for extracting the dataset from batch files
def unpickle(file):
    import pickle
    with open(file,'rb') as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict

#function for putting the images in the right shape
def reshape_and_transpose(x):
    
    return x.reshape((len(x),3,32,32)).transpose(0,2,3,1)

#extracting the dataset
dictionary1 = unpickle("data_batch_1")
dictionary2 = unpickle("data_batch_2")
dictionary3 = unpickle("data_batch_3")
dictionary4 = unpickle("data_batch_4")
dictionary5 = unpickle("data_batch_5")
dictionary_test = unpickle("test_batch")

x1 = dictionary1[b'data']
x2 = dictionary2[b'data']
x3 = dictionary3[b'data']
x4 = dictionary4[b'data']
x5 = dictionary5[b'data']
testData=dictionary_test[b'data']


label1 = np.array(dictionary1[b'labels'])
label2 = np.array(dictionary2[b'labels'])
label3 = np.array(dictionary3[b'labels'])
label4 = np.array(dictionary4[b'labels'])
label5 = np.array(dictionary5[b'labels'])
testLabels= np.array(dictionary_test[b'labels'])

#function for normalizing the images
def normalize(x):    
    return x/255

#function for one hot encoding labels
def one_hot_encode(x):
    z = np.zeros((len(x), 10))
    z[list(np.indices((len(x),))) + [x]] = 1
    return z
    
    
trainData = np.concatenate((x1,x2,x3,x4,x5),axis=0)
trainData = reshape_and_transpose(trainData)
trainData = normalize(trainData)

trainLabels = np.concatenate((label1,label2,label3,label4,label5),axis=0)
trainLabels = one_hot_encode(trainLabels)

testData = reshape_and_transpose(testData)
testData = normalize(testData)

testLabels = one_hot_encode(testLabels)

dataLength = len(trainData)
testLength = len(testData)

print("\nTrain Set and Test Set built!\n")


#deleting old TensorFlow graphs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

######################
#BUILDING THE NETWORK#
######################
    
'''
This is a CNN. It has 2 convolutional layers, with 64 and 32 filters,
respectively. The size of the sliding windows is 5 for each conv layer.
The padding is setted to 'SAME': this means that the output of the
conv layers has the same height and width of the input. The pooling
layers have a sliding window of size 2 and a strides of 2. This means
that the pooling layers divide the height and width by 2, having a
final flattened output of shape (height/4)*(width/4)*(conv2_fmaps).
The weights are initialized with a truncated normal ditribution,
with stddev=0.03. Biases with stddev = 1 are added to each conv layer.
After the convolutional layers the net has 2 fully connected layers,
with 64 and 32 neurons, respectively. The weights and biases are
computed as before (in this case the biases have stddev = 0.01). 
Dropout is performed after every fully connected layers, with rate = 0.5. 
The activation function used is the "relu" function.
The learning algorith is the Adam Optimizer.
'''

height = 32
width = 32
channels = 3
n_inputs = height*width*channels

#parameters of convolutional layers
conv1_fmaps = 64
conv1_ksize = 5
conv2_fmaps = 32
conv2_ksize = 5

#parameters of fully connected network and outputs
n_fc1 = 64
n_fc2 = 32
n_outputs = 10

#resetting old graphs pending
reset_graph()



with tf.name_scope("inputs"):
    
    x = tf.placeholder(tf.float32, shape=[None, 32,32,3], name = "X")
    
    x_reshaped = tf.reshape(x, shape=[-1,height,width, channels])
    
    y = tf.placeholder(tf.float32, shape = [None,10], name = "y")
    
    
with tf.name_scope("conv1"):
    
    conv_filt_shape = [conv1_ksize, conv1_ksize, channels,
                      conv1_fmaps]

    # initialize weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                                      name='W1')
    bias = tf.Variable(tf.truncated_normal([conv1_fmaps]), name='b1')

    
    conv1 = tf.nn.conv2d(x_reshaped, weights ,[1,1,1,1],padding="SAME", name="conv1")
    
    conv1 += bias
    
    conv1 = tf.nn.relu(conv1)
    

with tf.name_scope("pool1"):
    
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding="VALID")

    
with tf.name_scope("conv2"):
    
    conv_filt_shape2 = [conv2_ksize, conv2_ksize,conv1_fmaps,
                        conv2_fmaps]

    # initialise weights and bias for the filter
    weights2 = tf.Variable(tf.truncated_normal(conv_filt_shape2, stddev=0.03),
                                      name='W2')
    bias2 = tf.Variable(tf.truncated_normal([conv2_fmaps]), name='b2')

    
    conv2 = tf.nn.conv2d(pool1, weights2 ,[1,1,1,1],padding="SAME", name="conv2")
    
    conv2 += bias2
    
    conv2 = tf.nn.relu(conv2)
    


with tf.name_scope("pool2"):
    
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    
    #we flatten the pooling output in order
    #to use it in the fully connected layer
    pool2_flat = tf.reshape(pool2, shape=[-1 , 8 * 8 * 32])
    
with tf.name_scope("fc1"):
    
    wd1 = tf.Variable(tf.truncated_normal([8 * 8 * 32 , n_fc1], stddev=0.03), name='wd1')
    bd1 = tf.Variable(tf.truncated_normal([n_fc1], stddev=0.01), name='bd1')
    fc1 = tf.matmul(pool2_flat, wd1) + bd1
    fc1 = tf.layers.dropout(fc1,rate=0.5);
    fc1 = tf.nn.relu(fc1)
    
with tf.name_scope("fc2"):
    
    wd2 = tf.Variable(tf.truncated_normal([n_fc1, n_fc2], stddev=0.03), name='wd2')
    bd2 = tf.Variable(tf.truncated_normal([n_fc2], stddev=0.01), name='bd2')
    fc2 = tf.matmul(fc1, wd2) + bd2
    fc2 = tf.layers.dropout(fc2,rate=0.5);
    fc2 = tf.nn.relu(fc2)

    
with tf.name_scope("output"):
    
    wd3 = tf.Variable(tf.truncated_normal([n_fc2, 10], stddev=0.03), name='wd2')
    bd3 = tf.Variable(tf.truncated_normal([10], stddev=0.01), name='bd2')
    output = tf.matmul(fc2, wd3) + bd3
    y_proba = tf.nn.softmax(output, name="Y_proba")

with tf.name_scope("train"):
    
    xentropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))
    
    optimizer = tf.train.AdamOptimizer().minimize(xentropy)


with tf.name_scope("eval"):
    
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_proba, 1))
    
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init"):
    
    init = tf.global_variables_initializer()
    
##########################
#TRAINING & TESTING PHASE#
##########################
    
'''
The net is trained with all dataset, divided by batches
of dimension batch_size. The accuracy on both train set
and test set is computed on random batch of size batch_size
taken from the dataset.
'''
    
n_epochs = 1000
batch_size = 1000

#function for taking a random batch from trainSet or testSet
def random_batch(length):
    idx = np.random.choice(length, size = batch_size, replace=False)
    
    x_batch = trainData[idx,:,:,:]
    y_batch = trainLabels[idx,:]
    return x_batch, y_batch



with tf.Session() as sess:
    
    # initialise the variables
    sess.run(init)
    
    total_batch = int(dataLength / batch_size)
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        for i in range(total_batch):
             
            x_batch = trainData[i*batch_size:i*batch_size+batch_size,:]            
             
            y_batch = trainLabels[i*batch_size:i*batch_size+batch_size]
             
            _, c = sess.run([optimizer, xentropy], 
                            feed_dict={x: x_batch, y: y_batch})
            epoch_loss += c
        
        rx1_batch , ry1_batch = random_batch(dataLength)
        
        rx2_batch , ry2_batch = random_batch(testLength)
        
        tr_acc = sess.run(accuracy,feed_dict = {x:rx1_batch,y:ry1_batch})
        
        te_acc = sess.run(accuracy,feed_dict = {x:rx2_batch,y:ry2_batch})
        
        print("Epoch: ", epoch+1, "; Loss: ", epoch_loss, "; Train Accuracy: ",
              tr_acc,"; Test Accuracy: ",te_acc)
        
    print("\nTraining complete!")
    
    print("\nFinal Test Accuracy: ",te_acc)

