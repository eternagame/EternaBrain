# -*- coding: utf-8 -*-
"""
Created on Thu Jun 8

@author: rohankoodli
"""

import numpy as np
import os
from readData import read_movesets_pid, read_structure
from encodeRNA import encode_movesets, encode_structure
import tensorflow as tf
from tensorflow.contrib import rnn
import pickle
# enc0 = np.array([[[[1,2,3,4],[0,1,0,1],[-33,0,0,0]],[[1,2,3,4],[0,1,1,0],[-23,0,0,0]]],[[[3,3,3,3],[0,0,0,0],[2,0,0,0]],[[1,1,1,0],[1,0,1,0],[-23,0,0,0]]]])
# ms0 = np.array([[[2,1],[4,3]],[[1,6],[2,9]]])
# enc = np.array([[[1,2,3,4],[0,1,0,1],[1,1,1,1],[-3,0,0,0]],[[4,3,2,1],[1,0,1,0],[0,0,0,0],[9,0,0,0]]])
# out = np.array([[4,2],[3,3]])


real_X = pickle.load(open(os.getcwd()+'/pickles/X-6892348','rb'))
real_y = pickle.load(open(os.getcwd()+'/pickles/y-6892348','rb'))

real_X_9 = np.array(real_X[0:1500]).reshape([-1,340])
real_y_9 = np.array(real_y[0:1500])
test_real_X = np.array(real_X[1500:1600]).reshape([-1,340])
test_real_y = np.array(real_y[1500:1600])

#real_X_9, test_real_X, real_y_9, test_real_y = np.array(train_test_split(real_X[0:500],real_y[0:500],test_size=0.2))
#real_X_9, test_real_X, real_y_9, test_real_y = np.array(real_X_9).reshape([-1,340]), np.array(test_real_X).reshape([-1,340]), np.array(real_y_9), np.array(test_real_y)

# enc0 = np.array([[[1,2,3,4],[0,1,0,1],[-33,0,0,0],[1,1,1,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]]])
# ms0 = np.array([[1,6],[2,7],[2,7],[2,7],[2,7],[2,7],[2,7],[2,7],[2,7]])
# ms0 = np.array([[1,0,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0]]) # just base
#
# test_enc0 = np.array([[[2,3,3,2],[0,0,0,0],[6,0,0,0],[0,0,1,1]],[[1,2,3,4],[0,1,0,1],[-33,0,0,0],[1,1,1,1]]])
# test_ms0 = np.array([[4,20],[3,15]])
# test_ms0 = np.array([[0,0,0,1],[1,0,0,0]]) # just base

num_epochs = 10
n_classes = 4
batch_size = 1500 # same as training size
chunk_size = 4
n_chunks = 85
rnn_size = 128
TRAIN_KEEP_PROB = 0.5

x = tf.placeholder('float',[None,n_chunks,chunk_size]) # 16 with enc0
y = tf.placeholder('float')
keep_prob = tf.placeholder('float')

# enc = enc0.reshape([-1,16])
# ms = ms0#.reshape([-1,4])
#
# test_enc = test_enc0.reshape([-1,16])
# test_ms = test_ms0
#e1 = tf.reshape(enc0,[])

def recurrentNeuralNet(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
            'biases':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1,chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size)

    lstm_cell = rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)

    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    ol = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return ol


def train(x):
    prediction = recurrentNeuralNet(x)
    #print prediction
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost) # learning rate = 0.001

    # cycles of feed forward and backprop

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            epoch_loss = 0
            for _ in range(int(real_X_9.shape[0])):#mnist.train.num_examples/batch_size)): # X.shape[0]
                epoch_x,epoch_y = real_X_9,real_y_9 #mnist.train.next_batch(batch_size) # X,y
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))
                _,c = sess.run([optimizer,cost],feed_dict={x:epoch_x,y:epoch_y, keep_prob: TRAIN_KEEP_PROB})
                epoch_loss += c
            print 'Epoch', epoch + 1, 'completed out of', num_epochs, '\nLoss:',epoch_loss,'\n'

        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))

        print 'Accuracy', accuracy.eval(feed_dict={x:test_real_X.reshape((-1, n_chunks, chunk_size)), y:test_real_y, keep_prob: 1.0}) #X, y #mnist.test.images, mnist.test.labels

        # data for tensorboard
        writer = tf.summary.FileWriter(os.getcwd()+'/tensorboard/baseRNN-1500-100-128-10')
        writer.add_graph(sess.graph)
        '''
        Run this:
        tensorboard --logdir=tensorboard/baseRNN-SPECIFICATIONS --debug
        '''


train(x)
