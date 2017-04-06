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


structurepath = os.getcwd() + '/movesets/puzzle-structure-data.txt'
movesetpath = os.getcwd() + '/movesets/move-set-11-14-2016.txt'

ms, users = read_movesets_pid(movesetpath,6892348)
encoded = np.matrix(encode_movesets(ms))


structures = read_structure(structurepath)
structures_encoded = encode_structure(structures['structure'][19684])

n_nodes_hl1 = 500 # hidden layer 1 
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100 # load 100 features at a time 

x = tf.placeholder('float',[None,4716]) # 4716 w/ pid 6892348
y = tf.placeholder('float')


def neuralNet(data):
    hl_1 = {'weights':tf.Variable(tf.random_normal([4716, n_nodes_hl1])),
            'biases':tf.Variable(tf.random_normal(n_nodes_hl1))}

    hl_2 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
            'biases':tf.Variable(tf.random_normal(n_nodes_hl2))}

    hl_3 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
            'biases':tf.Variable(tf.random_normal(n_nodes_hl3))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
            'biases':tf.Variable(tf.random_normal([n_classes]))}
    
    l1 = tf.add(tf.matmul(data, hl_1['weights']) + hl_1['biases'])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1, hl_2['weights']) + hl_2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hl_3['weights']) + hl_3['biases'])
    l3 = tf.nn.relu(l3)

    ol = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
    
    return ol




