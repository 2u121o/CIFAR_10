import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np

###################################
#FIRST NEURAL NETWORK FOR CIFAR-10#
###################################

#####################
#OPENING THE DATASET#
#####################

#importing files of dataset
def unpickle(file):
    import pickle
    with open(file,'rb') as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict

#we transform the scalar labels into 1D tensor which has only one component equal to one
def oneHotEncode(vector):
    new = np.zeros((len(vector),10))
    for i in range(len(vector)):
        new[i,vector[i]] = 1
    return new

#extracting the train data
dictionary1 = unpickle("data_batch_1")
dictionary2 = unpickle("data_batch_2")
dictionary3 = unpickle("data_batch_3")
dictionary4 = unpickle("data_batch_4")
dictionary5 = unpickle("data_batch_5")

x1 = dictionary1[b'data']
x2 = dictionary2[b'data']
x3 = dictionary3[b'data']
x4 = dictionary4[b'data']
x5 = dictionary5[b'data']

#building the full trainSet
trainData = np.concatenate((x1,x2,x3,x4,x5),axis=0)


#length of the dataset
dataLength = len(trainData)

#the labels are lists, so we convert them in numpy arrays
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

#length of the test set
testLength = len(testData)

#building the full testLabels
testLabels = np.array(dictionaryTest[b'labels'])
testLabels = oneHotEncode(testLabels)

print("\nData Ready!\n")


######################
#BUILDING THE NETWORK#
######################

"""

This is a simplified network with 5 fully connected layers,
random normalized weights and biases with mean = 0 and 
stddev = 0.01. We adopted the <Adam optimizer>
as algorithm of minimization of the error
with learning rate standard = 0.001

"""

#dimension of the input
height = 32
width = 32
channels = 3 #RGB
n_bytes = height*width*channels

#defining the number of neurons of the three layers
n_nodes_hl1 = 500
n_nodes_hl2 = 400
n_nodes_hl3 = 300
n_nodes_hl4 = 200
n_nodes_hl5 = 100

#number of classes
n_classes = 10

#dimension of the batch
batch_size = 100

#placeholders

x = tf.placeholder('float', [None,n_bytes])
y = tf.placeholder('float')

def neural_network(input_data):
    
    hl1 = {'weights':tf.Variable(tf.random_normal([n_bytes,n_nodes_hl1])),
           'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    
    
    
    hl2 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
           'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    
    
    hl3 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
           'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    

    hl4 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_nodes_hl4])),
		   'biases': tf.Variable(tf.random_normal([n_nodes_hl4]))}
    
    
    hl5 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl4,n_nodes_hl5])),
		   'biases': tf.Variable(tf.random_normal([n_nodes_hl5]))}
    
    
    output = {'weights':tf.Variable(tf.random_normal([n_nodes_hl5,n_classes])),
           'biases': tf.Variable(tf.random_normal([n_classes]))}
         
    
    l1 = tf.add(tf.matmul(input_data,hl1['weights']),hl1['biases'])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1,hl2['weights']),hl2['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2,hl3['weights']),hl3['biases'])
    l3 = tf.nn.relu(l3)
    
    l4 = tf.add(tf.matmul(l3,hl4['weights']),hl4['biases'])
    l4 = tf.layers.dropout(l4, rate = 0.25)
    l4 = tf.nn.relu(l4)
    
    l5 = tf.add(tf.matmul(l4,hl5['weights']),hl5['biases'])
    l5 = tf.layers.dropout(l5, rate = 0.25)
    l5 = tf.nn.relu(l5)
    
    output = tf.add(tf.matmul(l5,output['weights']),output['biases'])
    return output


#we extract a random batch from the train set
def random_batch():
    
    index_list = np.random.choice(dataLength, size = batch_size, replace=False)
    x_batch = trainData[index_list,:]
    y_batch = trainLabels[index_list,:]
    return x_batch, y_batch

#we eztract a random batch from the test set
def random_batch_test():
    
    index_list = np.random.choice(testLength, size = batch_size, replace=False)
    x_batch = testData[index_list,:]
    y_batch = testLabels[index_list,:]
    return x_batch, y_batch

#Starting optimization
def train_op(data):
    
    result = neural_network(data)
    
    
    proba = tf.nn.softmax(result)
    
    #total error
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=result,labels=y))
    
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    #boolean vector
    
    correct = tf.equal(tf.argmax(proba,1),tf.argmax(y,1))
    
    
    accuracy = tf.reduce_mean(tf.cast(correct,'float'))
    
    #defining the epoch
    n_epoch = 20
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        #feed the net with deterministic batches from the train set
        #the train is scanned from begin to end (50000 images) at every epoch
        for epoch in range(n_epoch):
            epoch_loss = 0
            for j in range(int(dataLength/batch_size)):
                i=j
                x_b=trainData[i*batch_size:i*batch_size+batch_size]
                y_b=trainLabels[i*batch_size:i*batch_size+batch_size]
                j,c=sess.run([optimizer,cost], feed_dict = {x:x_b,y:y_b})
                epoch_loss+=c
                
            #we print the descent error between the prediction 
            #of the network and the real value of the labels
            print('Epoch:',epoch+1," loss:",epoch_loss)
            
        
            #compute training accuracy
            random_x , random_y = random_batch()
            train_acc = sess.run(accuracy, feed_dict = {x: random_x, y: random_y})
        
            print("Training accuracy: ", train_acc*100,"%")
            
            #testing phase
            random_x , random_y = random_batch_test()
            test_acc = sess.run(accuracy, feed_dict ={x:random_x,y:random_y})
            print('Test accuracy: ',test_acc)
            

print("\nStarting training & testing phase...\n")
train_op(x)

