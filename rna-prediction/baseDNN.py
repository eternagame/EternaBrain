# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 21:49:27 2017

@author: rohankoodli
"""

import numpy as np
import os
import tensorflow as tf
import pickle
#from sklearn.cross_validation import train_test_split

# enc0 = np.array([[[[1,2,3,4],[0,1,0,1],[-33,0,0,0]],[[1,2,3,4],[0,1,1,0],[-23,0,0,0]]],[[[3,3,3,3],[0,0,0,0],[2,0,0,0]],[[1,1,1,0],[1,0,1,0],[-23,0,0,0]]]])
# ms0 = np.array([[[2,1],[4,3]],[[1,6],[2,9]]])
# enc = np.array([[[1,2,3,4],[0,1,0,1],[1,1,1,1],[-3,0,0,0]],[[4,3,2,1],[1,0,1,0],[0,0,0,0],[9,0,0,0]]])
# out = np.array([[4,2],[3,3]])
features6892348 = pickle.load(open(os.getcwd()+'/pickles/X-6892348-dev','rb'))
labels6892348 = pickle.load(open(os.getcwd()+'/pickles/y-6892348-dev','rb'))
features6892346 = pickle.load(open(os.getcwd()+'/pickles/X-6892346','rb'))
labels6892346 = pickle.load(open(os.getcwd()+'/pickles/y-6892346','rb'))

real_X = features6892346+features6892348
real_y = labels6892346+labels6892348

TRAIN_KEEP_PROB = 1.0
TEST_KEEP_PROB = 1.0
learning_rate = 0.0001
#tb_path = '/tensorboard/baseDNN-500-10-10-50-100'

train = 56000
test = 200
num_nodes = 500

testtest = np.array(real_X[train:train+test]).reshape([-1,340])

real_X_9 = np.array(real_X[0:train]).reshape([-1,340])
real_y_9 = np.array(real_y[0:train])
test_real_X = np.array(real_X[train:train+test]).reshape([-1,340])
test_real_y = np.array(real_y[train:train+test])

#real_X_9, test_real_X, real_y_9, test_real_y = np.array(train_test_split(real_X[0:train],real_y[0:train],test_size=0.001))
#real_X_9, test_real_X, real_y_9, test_real_y = np.array(real_X_9).reshape([-1,340]), np.array(test_real_X).reshape([-1,340]), np.array(real_y_9), np.array(test_real_y)

# enc0 = np.array([[[1,2,3,4],[0,1,0,1],[-33,0,0,0],[1,1,1,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]]])
# ms0 = np.array([[1,6],[2,7],[2,7],[2,7],[2,7],[2,7],[2,7],[2,7],[2,7]])
# ms0 = np.array([[1,0,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0]]) # just base
#
# test_enc0 = np.array([[[2,3,3,2],[0,0,0,0],[6,0,0,0],[0,0,1,1]],[[1,2,3,4],[0,1,0,1],[-33,0,0,0],[1,1,1,1]]])
# test_ms0 = np.array([[4,20],[3,15]])
# test_ms0 = np.array([[0,0,0,1],[1,0,0,0]]) # just base

n_nodes_hl1 = num_nodes # hidden layer 1
n_nodes_hl2 = num_nodes
n_nodes_hl3 = num_nodes
n_nodes_hl4 = num_nodes
n_nodes_hl5 = num_nodes
n_nodes_hl6 = num_nodes
n_nodes_hl7 = num_nodes
n_nodes_hl8 = num_nodes
n_nodes_hl9 = num_nodes
n_nodes_hl10 = num_nodes

n_classes = 4
batch_size = 100 # load 100 features at a time

x = tf.placeholder('float',[None,340]) # 216 with enc0
y = tf.placeholder('float')
keep_prob = tf.placeholder('float')

# enc = enc0.reshape([-1,16])
# ms = ms0#.reshape([-1,4])
#
# test_enc = test_enc0.reshape([-1,16])
# test_ms = test_ms0

#e1 = tf.reshape(enc0,[])

def neuralNet(data):
    hl_1 = {'weights':tf.Variable(tf.random_normal([340, n_nodes_hl1]),name='Weights'),
            'biases':tf.Variable(tf.random_normal([n_nodes_hl1]),name='Biases')}

    hl_2 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2]),name='Weights'),
            'biases':tf.Variable(tf.random_normal([n_nodes_hl2]),name='Biases')}

    hl_3 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3]),name='Weights'),
            'biases':tf.Variable(tf.random_normal([n_nodes_hl3]),name='Biases')}

    hl_4 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4]),name='Weights'),
            'biases':tf.Variable(tf.random_normal([n_nodes_hl4]),name='Biases')}

    hl_5 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl4, n_nodes_hl5]),name='Weights'),
            'biases':tf.Variable(tf.random_normal([n_nodes_hl5]),name='Biases')}

    hl_6 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl5, n_nodes_hl6]),name='Weights'),
            'biases':tf.Variable(tf.random_normal([n_nodes_hl6]),name='Biases')}

    hl_7 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl6, n_nodes_hl7]),name='Weights'),
            'biases':tf.Variable(tf.random_normal([n_nodes_hl7]),name='Biases')}

    hl_8 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl7, n_nodes_hl8]),name='Weights'),
            'biases':tf.Variable(tf.random_normal([n_nodes_hl8]),name='Biases')}

    hl_9 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl8, n_nodes_hl9]),name='Weights'),
            'biases':tf.Variable(tf.random_normal([n_nodes_hl9]),name='Biases')}

    hl_10 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl9, n_nodes_hl10]),name='Weights'),
            'biases':tf.Variable(tf.random_normal([n_nodes_hl10]),name='Biases')}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl10, n_classes]),name='Weights-outputlayer'),
            'biases':tf.Variable(tf.random_normal([n_classes]),name='Biases-outputlayer')}

    l1 = tf.add(tf.matmul(data, hl_1['weights']), hl_1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hl_2['weights']), hl_2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hl_3['weights']), hl_3['biases'])
    l3 = tf.nn.relu(l3)

    l4 = tf.add(tf.matmul(l3, hl_4['weights']), hl_4['biases'])
    l4 = tf.nn.relu(l4)

    l5 = tf.add(tf.matmul(l4, hl_5['weights']), hl_5['biases'])
    l5 = tf.nn.relu(l5)

    l6 = tf.add(tf.matmul(l5, hl_6['weights']), hl_6['biases'])
    l6 = tf.nn.relu(l6)

    l7 = tf.add(tf.matmul(l6, hl_7['weights']), hl_7['biases'])
    l7 = tf.nn.relu(l7)

    l8 = tf.add(tf.matmul(l7, hl_8['weights']), hl_8['biases'])
    l8 = tf.nn.relu(l8)

    l9 = tf.add(tf.matmul(l8, hl_9['weights']), hl_9['biases'])
    l9 = tf.nn.relu(l9)

    l10 = tf.add(tf.matmul(l9, hl_10['weights']), hl_10['biases'])
    l10 = tf.nn.relu(l10)

    dropout = tf.nn.dropout(l10,keep_prob)
    ol = tf.matmul(dropout, output_layer['weights']) + output_layer['biases']

    tf.summary.histogram('weights-hl_1',hl_1['weights'])
    tf.summary.histogram('biases-hl_1',hl_1['biases'])
    tf.summary.histogram('act-hl_1',l1)

    tf.summary.histogram('weights-hl_2',hl_2['weights'])
    tf.summary.histogram('biases-hl_2',hl_2['biases'])
    tf.summary.histogram('act-hl_2',l2)

    tf.summary.histogram('weights-hl_3',hl_3['weights'])
    tf.summary.histogram('biases-hl_3',hl_3['biases'])
    tf.summary.histogram('act-hl_3',l3)

    tf.summary.histogram('weights-hl_4',hl_4['weights'])
    tf.summary.histogram('biases-hl_4',hl_4['biases'])
    tf.summary.histogram('act-hl_4',l4)

    tf.summary.histogram('weights-hl_5',hl_5['weights'])
    tf.summary.histogram('biases-hl_5',hl_5['biases'])
    tf.summary.histogram('act-hl_5',l5)

    tf.summary.histogram('weights-hl_6',hl_6['weights'])
    tf.summary.histogram('biases-hl_6',hl_6['biases'])
    tf.summary.histogram('act-hl_6',l6)

    tf.summary.histogram('weights-hl_7',hl_7['weights'])
    tf.summary.histogram('biases-hl_7',hl_7['biases'])
    tf.summary.histogram('act-hl_7',l7)

    tf.summary.histogram('weights-hl_8',hl_8['weights'])
    tf.summary.histogram('biases-hl_8',hl_8['biases'])
    tf.summary.histogram('act-hl_8',l8)

    tf.summary.histogram('weights-hl_9',hl_9['weights'])
    tf.summary.histogram('biases-hl_9',hl_9['biases'])
    tf.summary.histogram('act-hl_9',l9)

    tf.summary.histogram('weights-hl_10',hl_10['weights'])
    tf.summary.histogram('biases-hl_10',hl_10['biases'])
    tf.summary.histogram('act-hl_10',l10)

    return ol


def train(x):
    prediction = neuralNet(x)
    #print prediction
    with tf.name_scope('cross_entropy'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
        tf.summary.scalar('cross_entropy',cost)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost) # learning rate = 0.001

    with tf.name_scope('accuracy'):
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        tf.summary.scalar('accuracy',accuracy)

    # cycles of feed forward and backprop
    num_epochs = 30

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        merged_summary = tf.summary.merge_all()
        #writer = tf.summary.FileWriter(os.getcwd()+tb_path)
        #writer.add_graph(sess.graph)

        for epoch in range(num_epochs):
            epoch_loss = 0
            for i in range(int(real_X_9.shape[0])/batch_size):#mnist.train.num_examples/batch_size)): # X.shape[0]
                randidx = np.random.choice(real_X_9.shape[0], batch_size, replace=False)
                epoch_x,epoch_y = real_X_9[randidx,:],real_y_9[randidx,:] #mnist.train.next_batch(batch_size) # X,y
                j,c = sess.run([optimizer,cost],feed_dict={x:epoch_x,y:epoch_y,keep_prob:TRAIN_KEEP_PROB})
                if i == 0:
                    [ta] = sess.run([accuracy],feed_dict={x:epoch_x,y:epoch_y,keep_prob:TRAIN_KEEP_PROB})
                    print 'Train Accuracy', ta
                # if i % 5 == 0:
                #     s = sess.run(merged_summary,feed_dict={x:epoch_x,y:epoch_y,keep_prob:TRAIN_KEEP_PROB})
                #     writer.add_summary(s,i)

                epoch_loss += c
            print '\n','Epoch', epoch + 1, 'completed out of', num_epochs, '\nLoss:',epoch_loss


        print '\n','Train Accuracy', accuracy.eval(feed_dict={x:real_X_9, y:real_y_9, keep_prob:TRAIN_KEEP_PROB})
        print '\n','Test Accuracy', accuracy.eval(feed_dict={x:test_real_X, y:test_real_y, keep_prob:1.0}) #X, y #mnist.test.images, mnist.test.labels

        #print 'Prediction',sess.run(prediction, feed_dict={x:testtest, keep_prob:1})
        #print 'Prediction',sess.run(tf.argmax(prediction,1), feed_dict={x:testtest, keep_prob:1})
        #print test_real_y
        # correct_list = []
        # for i in range(len(sess.run(tf.argmax(prediction,1), feed_dict={x:testtest, keep_prob:1}))):
        #     if list(test_real_y[i]).index(1) == sess.run(tf.argmax(prediction,1), feed_dict={x:testtest, keep_prob:1})[i]:
        #         correct_list.append(True)
        #     else:
        #         correct_list.append(False)
        # print correct_list

        '''
        saver = tf.train.Saver()
        saver.save(sess,os.getcwd()+'/models/baseDNN')
        '''

        '''
        Run this:
        tensorboard --logdir=tensorboard/baseDNN-SPECIFICATIONS --debug
        '''
    '''
    sess2 = tf.Session()
    print sess2.run(tf.argmax(y,1), feed_dict={x: np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).reshape([-1,340])})
    sess2.close()

    sess2 = tf.Session()
    with sess2.as_default():

    sess2.close()
    '''
train(x)
