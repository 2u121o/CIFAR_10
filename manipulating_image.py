# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 23:26:49 2018

@author: andre
"""

#######################
#PREPROCESSING STATICO#
#######################

import tensorflow as tf
import numpy as np

#extracting dataset
def unpickle(file):
    import pickle
    with open(file,'rb') as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict

#we transform the scalar labels into 1D tensor which has only 
#one component equal to one
def oneHotEncode(vector):
    new = np.zeros((len(vector),10),dtype='float32')
    for i in range(len(vector)):
        new[i,vector[i]] = 1
    return new

#for normalize the images
def normalize(vector):
    return vector/255

#transform a raw-like np array into a RGB image
def reshapeAndTranspose(img):
    return img.reshape(len(img),3,32,32).transpose(0,2,3,1)


#extracting the train data
dictionary1 = unpickle("data_batch_1")
dictionary2 = unpickle("data_batch_2")
dictionary3 = unpickle("data_batch_3")
dictionary4 = unpickle("data_batch_4")
dictionary5 = unpickle("data_batch_5")

#x_i are already numpy arrays (shape = [50000,3072])
x1 = dictionary1[b'data']
x2 = dictionary2[b'data']
x3 = dictionary3[b'data']
x4 = dictionary4[b'data']
x5 = dictionary5[b'data']

#building the full trainSet
trainData = np.concatenate((x1,x2,x3,x4,x5),axis=0)

trainData = reshapeAndTranspose(trainData)

trainData = normalize(trainData)

#casting images to float32
trainData = trainData.astype('float32')


#length of the dataset
dataLength = len(trainData)



#the labels are lists, so we convert them in numpy arrays  (shape = [10000,1])
labels1 = np.array(dictionary1[b'labels'])
labels2 = np.array(dictionary2[b'labels'])
labels3 = np.array(dictionary3[b'labels'])
labels4 = np.array(dictionary4[b'labels'])
labels5 = np.array(dictionary5[b'labels'])

#building the full trainLabels
trainLabels = np.concatenate((labels1,labels2,labels3,labels4,labels5),axis = 0)

trainLabels = oneHotEncode(trainLabels)



#extracting the test data
dictionaryTest = unpickle("test_batch")

testData = dictionaryTest[b'data']

#modifying testData
testData = reshapeAndTranspose(testData)

testData = normalize(testData)

#casting test images to float32
testData = testData.astype('float32')

#length of the test data
testLength = len(testData)

#building the full testLabels
testLabels = np.array(dictionaryTest[b'labels'])


testLabels = oneHotEncode(testLabels)

'''

Now we have the train&test data&labels as float 32, 

'''
print("Data loaded!\n")

height = 32
width = 32
img_crop = 24
channels = 3

'''
Modifying the dataset with TensorFlow.Image libraries.
The images from train set have been cropped to 24x24,
randomly flipped from left to right and changed in hue,
contrast, saturation and brightness
'''


def modifyImageTrain(image_):
    
    image = image_
    
    # Randomly crop the input image.
    image = tf.random_crop(image, size=[img_crop, img_crop, channels])
    
    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)
            
    # Randomly adjust hue, contrast and saturation.
    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
    
    #Limit the pixels between [0,1] in case of overflow
    image = tf.minimum(image, 1.0)
    image = tf.maximum(image, 0.0)    
    
    return image

def allImagesTrain(images_):
    images = images_
    images = tf.map_fn(lambda image: modifyImageTrain(image), images)
    return images

'''
The test images have been randomply cropped to 24x24 with
TensorFlow.Image libraries
'''

def modifyImageTest(image_):
    
    image = image_
    
    # Randomly crop the test image to 24x24
    image = tf.image.resize_image_with_crop_or_pad(image,
                                                   target_height=img_crop,
                                                   target_width=img_crop)
    return image

def allImagesTest(images_):
    images = images_
    images = tf.map_fn(lambda image: modifyImageTest(image),images)
    return images

#Initializing final numpy arrays for dataset
trainSet = np.zeros((0,img_crop,img_crop,channels), dtype = 'float32')
testSet = np.zeros((0,img_crop,img_crop,channels), dtype = 'float32')


print("\nStarting modifying Train Set...\n")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch_size = 1000
    
    
    
    for i in range(int(dataLength/batch_size)):
        
        temp_ = allImagesTrain(trainData[i*batch_size:i*batch_size+batch_size])
        temp = sess.run(temp_)
        
        trainSet = np.append(trainSet,temp,axis=0)
        print("Completato: ",(i+1)/int(dataLength/batch_size)*100,"%")


sess.close()

np.save("FilePreprocess/trainSet",trainSet)
print('\nTrain Set saved!\n')

print("\nStarting modifying Test Set...\n")    
with tf.Session() as sess:
    
    batch_size = 5000
    
    sess.run(tf.global_variables_initializer())
    
    for i in range(int(testLength/batch_size)):
        temp_ = allImagesTest(testData[i*batch_size:i*batch_size+batch_size])
        temp = sess.run(temp_)
        testSet = np.append(testSet,temp,axis=0)
        print("Completato: ",(i+1)/int(testLength/batch_size)*100,"%") 

sess.close()

np.save("FilePreprocess/testSet",testSet)

print('\nTest Set saved!\n')

np.save("FilePreprocess/trainLabels",trainLabels)

np.save("FilePreprocess/testLabels",testLabels)

print("\nLabels saved!\n All stuffes done!\n")























