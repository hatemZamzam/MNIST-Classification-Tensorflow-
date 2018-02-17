# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 02:44:11 2018

@author: HatemZam
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
# import math
import time
from datetime import timedelta

# Conv. Layer 1
filter_size1 = 5
num_filters1 = 16

# Conv. Layer 2
filter_size2 = 5
num_filters2 = 36

# Fully Connected Layer
fc_size = 128             # neurons in fully conn.

# Load mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)
# check dataset lengths
print(len(data.train.labels))  # 55000
print(len(data.test.labels))   # 10000

data.test.cls = np.argmax(data.test.labels, axis=1)

# Data Dimensions
img_size = 28
img_size_flat = img_size*img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10

# see some images
sample = data.train.images[0].reshape(28, 28)
plt.imshow(sample, cmap = 'Greys')

# define functions to Create weights and biases tensor variables
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape = [length]))

# function to create new conv. layer
def new_conv_layer(input, num_input_channels, filter_size, num_filters, pooling = True):
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1,1,1,1], padding='SAME')
    layer += biases
    if pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
    layer = tf.nn.relu(layer)
    return layer, weights

# function to create the Flatten Layer
def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

# function to create fully connected layers
def new_fc_layer(input, inputs_num, outputs_num, relu_use=True):
    weights = new_weights(shape=[inputs_num, outputs_num])
    biases = new_biases(length=outputs_num)
    layer = tf.add(tf.matmul(input, weights), biases)
    if relu_use:
        layer = tf.nn.relu(layer)
    return layer

# Tensorflow Placeholders for inputs imgs, labels and class-num
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

# Create 1st convLayer
conv_layer1, conv_weights1 = new_conv_layer(input=x_image, num_input_channels=num_channels, filter_size=filter_size1, num_filters=num_filters1, pooling=True)
# Create 2nd convLayer
conv_layer2, conv_weights2 = new_conv_layer(input=conv_layer1, num_input_channels=num_filters1, filter_size=filter_size2, num_filters=num_filters2, pooling=True)

# Create the Flatten layer
flat_layer, num_features = flatten_layer(conv_layer2)

# Create 1st fully conn. layer
fc_layer1 = new_fc_layer(input=flat_layer, inputs_num=num_features, outputs_num=fc_size, relu_use=True)
# Create 2nd fully conn. layer
fc_layer2 = new_fc_layer(input=fc_layer1, inputs_num=fc_size, outputs_num=num_classes, relu_use=False)

# Apply softmax function to the output because of 10 classes-category
y_pred = tf.nn.softmax(fc_layer2)
y_pred_cls = tf.argmax(y_pred, axis=1)

# Optimization Cost-Function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc_layer2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

# Optimize (AdamOptimizer) => Gradient Descent
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# Measure Performance
correct_pred = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Tensorflow Session to Run
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Optimization iterations (batches)
batch_size = 64

total_iterations = 0

def optimize(iterations_num):
    global total_iterations
    start_time = time.time()
    for i in range(total_iterations, total_iterations+iterations_num):
        x_batch, y_true_batch = data.train.next_batch(batch_size)
        feed_dict_train = {x: x_batch, y_true: y_true_batch}
        sess.run(optimizer, feed_dict=feed_dict_train)
        #if i % 100 == 0:
        acc = sess.run(accuracy, feed_dict=feed_dict_train)
        msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
        print(msg.format(i + 1, acc))
    total_iterations += iterations_num
    end_time = time.time()
    diff_time = end_time-start_time
    print("Time usage: " + str(timedelta(seconds=int(round(diff_time)))))
    

optimize(785)

# Prediction--------------------------------------------
test_img = data.test.images[300]
plt.imshow(test_img.reshape(28, 28), cmap='Greys')
feed_dict_test = {x: [test_img]}
classify = sess.run(y_pred_cls, feed_dict=feed_dict_test)
print(classify)

# Save my model-----------------------------------------
saver = tf.train.Saver()
saver.save(sess=sess, save_path='data/my_mnist_tf_model.ckpt')

# Load model template-----------------------------------

# stackOverFlow :-
sess=tf.Session()    
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('data/my_mnist_tf_model.ckpt.meta')
#saver.restore(sess,tf.train.latest_checkpoint('./'))
# Access saved Variables directly
print(sess.run('bias:0'))
# This will print 2, which is the value of bias that we saved
# Now, let's access and create placeholders variables and
# create feed-dict to feed new data
graph = tf.get_default_graph()
y_p = graph.get_tensor_by_name("y_true")
y_true_cls = tf.argmax(y_true, axis=1)
#feed_dict ={w1:13.0,w2:17.0}
#Now, access the op that you want to run. 
test_img = data.test.images[300]
plt.imshow(test_img.reshape(28, 28), cmap='Greys')
feed_dict_test = {x: [test_img]}
classify = sess.run(y_pred_cls, feed_dict=feed_dict_test)
print(classify)
#This will print 60 which is calculated 

#tensorFlow webpage :-
tf.reset_default_graph()
# Create some variables.
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])
# Add ops to save and restore all the variables.
saver = tf.train.Saver()
# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Check the values of the variables
  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())

