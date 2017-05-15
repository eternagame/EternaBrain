# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 21:49:27 2017

@author: rohankoodli
"""

import numpy as np
import os
from readData import read_movesets_pid, read_structure
from encodeRNA import encode_movesets, encode_structure
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/Users/rohankoodli/Documents/MNIST/",one_hot=True)


enc = np.array([[[[1,2,3,4],[0,1,0,1],[-33,0,0,0]],[[1,2,3,4],[0,1,1,0],[-23,0,0,0]]],[[[3,3,3,3],[0,0,0,0],[2,0,0,0]],[[1,1,1,0],[1,0,1,0],[-23,0,0,0]]]])
ms = np.array([[[2,1],[4,3]],[[1,6],[2,9]]])

# enc = np.array([[[1,2,3,4],[0,1,0,1],[1,1,1,1],[-3,0,0,0]],[[4,3,2,1],[1,0,1,0],[0,0,0,0],[9,0,0,0]]])
# out = np.array([[4,2],[3,3]])

n_nodes_hl1 = 500 # hidden layer 1
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100 # load 100 features at a time


x = tf.placeholder('float',[None,2,2,3,4]) # 4716 w/ pid 6892348
y = tf.placeholder('float')

enc = enc.reshape([-1,24])


def neuralNet(data):
    hl_1 = {'weights':tf.Variable(tf.random_normal([2,2,3,4, n_nodes_hl1])),
            'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hl_2 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
            'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hl_3 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
            'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
            'biases':tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hl_1['weights']), hl_1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hl_2['weights']), hl_2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hl_3['weights']), hl_3['biases'])
    l3 = tf.nn.relu(l3)

    ol = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return ol


def train(x):
    prediction = neuralNet(x)
    print prediction
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost) # learning rate = 0.001

    # cycles of feed forward and backprop
    num_epochs = 15

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            epoch_loss = 0
            for _ in range(int(enc.shape[0])):#mnist.train.num_examples/batch_size)): # X.shape[0]
                epoch_x,epoch_y = enc,ms #mnist.train.next_batch(batch_size) # X,y
                _,c = sess.run([optimizer,cost],feed_dict={x:epoch_x,y:epoch_y})
                epoch_loss += c
            print 'Epoch', epoch + 1, 'completed out of', num_epochs, '\nLoss:',epoch_loss,'\n'

        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))

        print 'Accuracy', accuracy.eval({x:enc, y:ms}) #X, y #mnist.test.images, mnist.test.labels


train(x)
