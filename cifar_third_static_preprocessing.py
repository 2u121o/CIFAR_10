# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 17:50:46 2018

@author: andre
"""

####################################
#THIRD NEURAL NETWORK FOR CIFAR-10##
####################################

import numpy as np
import tensorflow as tf
import prettytensor as pt
import os


#defining the size of the image and other useful parameters
    
height = 24
width = 24
channels = 3
n_bytes = height*width*channels
n_classes = 10
NUM_EPOCHS = 25000




#list of categories
class_names = ['airplane',
 'car',
 'bird',
 'cat',
 'deer',
 'dog',
 'frog',
 'horse',
 'ship',
 'truck']

print("\nData ready!\n")

'''
Loading the preprocessed images and labels
'''

trainData = np.load('FilePreprocess/trainSet.npy')
trainLabels = np.load('FilePreprocess/trainLabels.npy')
testData = np.load('FilePreprocess/testSet.npy')
testLabels = np.load('FilePreprocess/testLabels.npy')

dataLength = len(trainData)

testLength = len(testData)

#Generating Test Label Scalar
testLabelScalar = np.zeros((testLength),dtype='float32')

for i in range(testLength):
    for j in range(n_classes):
        if (int(testLabels[i,j]) == 1):
            testLabelScalar[i] = j

print("\nData ready!\n")


######################
#BUILDING THE NETWORK#
######################

'''
This is a complex CNN. It was built with a sub-library
of tensorflow, named PrettyTensor. It has two convolutional
layers, each having 64 filters, a sliding window of 5 and
strides 2. After the first conv layer the net has
a batch normalization that reduce the noise into the image,
setting mean = 0 and stddev = 1.  After conv layers the net 
has max_pool, with kernel & strides = 2. After flattening 
2nd pool, it has  two fully connected layers with 256 and 
128 neurons. At the  end, of course, a softmax classifier.
'''



#deleting old TensorFlow graphs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


reset_graph()

x = tf.placeholder(tf.float32, shape = [None, height,width,channels], 
                   name = 'input')
y = tf.placeholder(tf.float32, shape=[None, n_classes],
                   name = 'labels')
y_cls_scalar = tf.argmax(y,axis=1)



def prettyNetwork(images, training):
    # Wrap the input images as a Pretty Tensor object
    x_pretty = pt.wrap(images)
    
    #special number by Pretty Tensor
    if training:
        phase = pt.Phase.train
    else:
        phase = pt.Phase.infer

    
    with pt.defaults_scope(activation_fn=tf.nn.relu, phase=phase):
        y_pred, loss = x_pretty.\
            conv2d(kernel=5, depth=64, name='layer_conv1', batch_normalize=True).\
            max_pool(kernel=2, stride=2).\
            conv2d(kernel=5, depth=64, name='layer_conv2').\
            max_pool(kernel=2, stride=2).\
            flatten().\
            fully_connected(size=256, name='layer_fc1').\
            fully_connected(size=128, name='layer_fc2').\
            softmax_classifier(num_classes=n_classes, labels=y)

    return y_pred, loss

def netVarScope(training):

    # creating new variables during training, and re-using during testing
    with tf.variable_scope('network', reuse = not training):
        
        images = x

        y_pred, loss = prettyNetwork(images=images, training=training)

    return y_pred, loss

#Define some parameters useful later for the optimization ops
global_step = tf.Variable(initial_value=0,
                          name='global_step', trainable=False)


#Training Phase, we want to calculate the loss, in order to minimize it later
_, loss = netVarScope(training=True)


#Defining an optimization algorithm, in this case AdamOptimizer,
#with learning rate = 1e-4
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, 
                                  global_step=global_step)


#Testing Phase, we want to calculate the prediction, of course
y_pred, _ = netVarScope(training=False)



#Testing Phase, compute the class of the prediction,
#passing by a one hot encoded vector to a scalar
#which represent the category of the predicted image
y_pred_cls = tf.argmax(y_pred, axis=1)



#Array of boolean, which has its components True if
#the prediction is correct
correct_prediction = tf.equal(y_pred_cls, y_cls_scalar)


#Number of right prediction, in terms of percentage
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#For saving parameters
saver = tf.train.Saver()

#Creating the Session
session = tf.Session()

#######################
#RESTORING CHECKPOINTS#
#######################

save_dir = 'checkpoints/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, 'cifar10_cnn')

try:
    print("Trying to restore last checkpoint ...")

    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)

    saver.restore(session, save_path=last_chk_path)

    print("Restored checkpoint from:", last_chk_path)

except:
    
    
    print("Failed to restore checkpoint. Initializing variables instead.")
    
    session.run(tf.global_variables_initializer())

train_batch_size = 64

#function for taking a random batch from training test
def random_batch():
    
    idx = np.random.choice(dataLength, size = train_batch_size, 
                           replace=False)
    
    x_batch = trainData[idx,:,:,:]
    y_batch = trainLabels[idx,:]
    return x_batch, y_batch

##################
#TRAINING THE NET#
##################




def launchNet(epochs):
        

    for i in range(epochs):
        
        x_batch, y_true_batch = random_batch()
        
        feed_dict_train = {x: x_batch,
                           y: y_true_batch}

        i_global, _ ,loss1= session.run([global_step, optimizer,
                                                       loss],
                                  feed_dict=feed_dict_train)
        
        
        print("Epoch: ",i+1,"; Loss: ",loss1)
    
        batch_acc = session.run(accuracy,
                                    feed_dict=feed_dict_train)

        print("Training Batch Accuracy: ",batch_acc*100,"%")
            
            
        #After every 1000 epochs we save the net parameter into a file    
        if (i_global % 1000 == 0) or (i == epochs - 1):
          
            saver.save(session,
                       save_path=save_path,
                       global_step=global_step)

            print("Saved checkpoint.")


batch_size = 128

#################
#TESTING THE NET#
#################

def testNet(images, labels, trueClasses):
    
    nImages = len(images)
    
    predicted = np.zeros(shape=nImages, dtype=np.int)
    
    
    for i in range(int(nImages/batch_size)):

        feed_dict = {x: images[i*batch_size:i*batch_size+batch_size, :],
                     y: labels[i*batch_size:i*batch_size+batch_size, :]}

        
        predicted[i*batch_size:i*batch_size+batch_size] = session.run(y_pred_cls, feed_dict=feed_dict)

    
    correct = (trueClasses == predicted)

    testAcc = correct.mean()
    
    print("Test Accuracy: ", testAcc*100,"%")


#Boolean variable that control the net ops to do 
train = 1


if (train==1):
    print("\nStarting training...\n\n\n")
    launchNet(epochs = NUM_EPOCHS)
    print("\nOptimization complete!\n")

else:    
    print("\nStarting testing...\n\n\n")
    testNet()
    print("\nTesting complete!\n")

#Closing the session
session.close()




