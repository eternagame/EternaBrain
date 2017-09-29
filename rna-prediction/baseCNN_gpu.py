# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 21:49:27 2017

@author: rohankoodli
"""

import numpy as np
import os
import tensorflow as tf
import pickle
from sklearn.cross_validation import train_test_split
from tf_funcs import average_gradients
#from matplotlib import pyplot as plt

# enc0 = np.array([[[[1,2,3,4],[0,1,0,1],[-33,0,0,0]],[[1,2,3,4],[0,1,1,0],[-23,0,0,0]]],[[[3,3,3,3],[0,0,0,0],[2,0,0,0]],[[1,1,1,0],[1,0,1,0],[-23,0,0,0]]]])
# ms0 = np.array([[[2,1],[4,3]],[[1,6],[2,9]]])
# enc = np.array([[[1,2,3,4],[0,1,0,1],[1,1,1,1],[-3,0,0,0]],[[4,3,2,1],[1,0,1,0],[0,0,0,0],[9,0,0,0]]])
# out = np.array([[4,2],[3,3]])

with open(os.getcwd()+'/movesets/teaching-puzzle-ids.txt') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]
content = [int(x) for x in content]
progression = [6502966,6502968,6502973,6502976,6502984,6502985,6502993, \
                6502994,6502995,6502996,6502997,6502998,6502999,6503000] # 6502957
content.extend(progression)
content.remove(6502966)
content.remove(6502976)
content.remove(6502984)

real_X = []
real_y = []
pids = []

for pid in content:
    feats = pickle.load(open(os.getcwd()+'/pickles/X-exp-loc-'+str(pid),'rb'))
    ybase = pickle.load(open(os.getcwd()+'/pickles/y-exp-base-'+str(pid),'rb'))
    yloc = pickle.load(open(os.getcwd()+'/pickles/y-exp-loc-'+str(pid),'rb'))
    for i in range(len(feats)):
        feats[i].append(yloc[i])
    real_X.extend(feats)
    real_y.extend(ybase)
    pids.append(feats)

print "Unpickled"

max_lens = []
for puzzle in pids:
    max_lens.append(len(puzzle[0][0]))

indxs = []
for i in range(len(max_lens)):
     if max_lens[i] < max(max_lens):
         indxs.append(i)

for i in indxs:
     if pids[i]:
         for j in pids[i]:
             for k in j:
                 k.extend([0]*(max(max_lens) - len(k)))

TRAIN_KEEP_PROB = 1.0
TEST_KEEP_PROB = 1.0
learning_rate = 0.0001
ne = 300
#tb_path = '/tensorboard/baseDNN-500-10-10-50-100'

train = 30000
test = 20
num_nodes = 250
len_puzzle = max(max_lens)

TF_SHAPE = 7 * len_puzzle

#testtest = np.array(real_X[train:train+test]).reshape([-1,TF_SHAPE])

real_X_9 = np.array(real_X[0:train]).reshape([-1,TF_SHAPE])
real_y_9 = np.array(real_y[0:train])
test_real_X = np.array(real_X[train:train+test]).reshape([-1,TF_SHAPE])
test_real_y = np.array(real_y[train:train+test])

print "Data prepped"

# real_X_9, test_real_X, real_y_9, test_real_y = np.array(train_test_split(real_X[0:train],real_y[0:train],test_size=0.01))
# real_X_9, test_real_X, real_y_9, test_real_y = np.array(real_X_9).reshape([-1,TF_SHAPE]), np.array(test_real_X).reshape([-1,TF_SHAPE]), np.array(real_y_9), np.array(test_real_y)

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

x = tf.placeholder('float',[None,TF_SHAPE],name="x_placeholder") # 216 with enc0
y = tf.placeholder('float',name='y_placeholder')
keep_prob = tf.placeholder('float',name='keep_prob_placeholder')

# enc = enc0.reshape([-1,16])
# ms = ms0#.reshape([-1,4])
#
# test_enc = test_enc0.reshape([-1,16])
# test_ms = test_ms0

#e1 = tf.reshape(enc0,[])

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

def convNeuralNet(x):
    weights = {'w_conv1':tf.get_variable('w_conv1',[7,7,1,2],initializer=tf.random_normal_initializer()),
               'w_conv2':tf.get_variable('w_conv2',[7,7,2,4],initializer=tf.random_normal_initializer()),
               'w_conv3':tf.get_variable('w_conv3',[7,7,4,8],initializer=tf.random_normal_initializer()),
               'w_conv4':tf.get_variable('w_conv4',[7,7,8,16],initializer=tf.random_normal_initializer()),
               'w_conv5':tf.get_variable('w_conv5',[7,7,16,32],initializer=tf.random_normal_initializer()),
               'w_conv6':tf.get_variable('w_conv6',[7,7,32,64],initializer=tf.random_normal_initializer()),
               'w_conv7':tf.get_variable('w_conv7',[7,7,64,128],initializer=tf.random_normal_initializer()),
               'w_conv8':tf.get_variable('w_conv8',[7,7,128,256],initializer=tf.random_normal_initializer()),
               'w_conv9':tf.get_variable('w_conv9',[7,7,256,512],initializer=tf.random_normal_initializer()),
               'w_fc1':tf.get_variable('w_fc1',[512,1024],initializer=tf.random_normal_initializer()),
               'w_fc2':tf.get_variable('w_fc2',[1024,2048],initializer=tf.random_normal_initializer()),
               'w_fc3':tf.get_variable('w_fc3',[2048,4096],initializer=tf.random_normal_initializer()),
               'out':tf.get_variable('w_out',[4096,n_classes],initializer=tf.random_normal_initializer())}

    biases = {'b_conv1':tf.get_variable('b_conv1',[2],initializer=tf.random_normal_initializer()),
              'b_conv2':tf.get_variable('b_conv2',[4],initializer=tf.random_normal_initializer()),
              'b_conv3':tf.get_variable('b_conv3',[8],initializer=tf.random_normal_initializer()),
              'b_conv4':tf.get_variable('b_conv4',[16],initializer=tf.random_normal_initializer()),
              'b_conv5':tf.get_variable('b_conv5',[32],initializer=tf.random_normal_initializer()),
              'b_conv6':tf.get_variable('b_conv6',[64],initializer=tf.random_normal_initializer()),
              'b_conv7':tf.get_variable('b_conv7',[128],initializer=tf.random_normal_initializer()),
              'b_conv8':tf.get_variable('b_conv8',[256],initializer=tf.random_normal_initializer()),
              'b_conv9':tf.get_variable('b_conv9',[512],initializer=tf.random_normal_initializer()),
              'b_fc1':tf.get_variable('b_fc1',[1024],initializer=tf.random_normal_initializer()),
              'b_fc2':tf.get_variable('b_fc2',[2048],initializer=tf.random_normal_initializer()),
              'b_fc3':tf.get_variable('b_fc3',[4096],initializer=tf.random_normal_initializer()),
              'out':tf.get_variable('b_out',[n_classes],initializer=tf.random_normal_initializer())}

    x = tf.reshape(x,shape=[-1,7,len_puzzle,1])

    conv1 = conv2d(x, weights['w_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = conv2d(conv1, weights['w_conv2'])
    conv2 = maxpool2d(conv2)

    conv3 = conv2d(conv2, weights['w_conv3'])
    conv3 = maxpool2d(conv3)

    conv4 = conv2d(conv3, weights['w_conv4'])
    conv4 = maxpool2d(conv4)

    conv5 = conv2d(conv4, weights['w_conv5'])
    conv5 = maxpool2d(conv5)

    conv6 = conv2d(conv5, weights['w_conv6'])
    conv6 = maxpool2d(conv6)

    conv7 = conv2d(conv6, weights['w_conv7'])
    conv7 = maxpool2d(conv7)

    conv8 = conv2d(conv7, weights['w_conv8'])
    conv8 = maxpool2d(conv8)

    conv9 = conv2d(conv8, weights['w_conv9'])
    conv9 = maxpool2d(conv9)

    fc1 = tf.reshape(conv9, [-1,512])
    fc1 = tf.nn.sigmoid(tf.add(tf.matmul(fc1,weights['w_fc1']),biases['b_fc1']))

    fc2 = tf.nn.sigmoid(tf.add(tf.matmul(fc1,weights['w_fc2']),biases['b_fc2']))

    fc3 = tf.nn.sigmoid(tf.add(tf.matmul(fc2,weights['w_fc3']),biases['b_fc3']))

    last = tf.nn.dropout(fc3,keep_prob)
    output = tf.add(tf.matmul(last, weights['out']), biases['out'], name='op7')

    return output

    # tf.summary.histogram('weights-hl_1',hl_1['weights'])
    # tf.summary.histogram('biases-hl_1',hl_1['biases'])
    # tf.summary.histogram('act-hl_1',l1)
    #
    # tf.summary.histogram('weights-hl_2',hl_2['weights'])
    # tf.summary.histogram('biases-hl_2',hl_2['biases'])
    # tf.summary.histogram('act-hl_2',l2)
    #
    # tf.summary.histogram('weights-hl_3',hl_3['weights'])
    # tf.summary.histogram('biases-hl_3',hl_3['biases'])
    # tf.summary.histogram('act-hl_3',l3)
    #
    # tf.summary.histogram('weights-hl_4',hl_4['weights'])
    # tf.summary.histogram('biases-hl_4',hl_4['biases'])
    # tf.summary.histogram('act-hl_4',l4)
    #
    # tf.summary.histogram('weights-hl_5',hl_5['weights'])
    # tf.summary.histogram('biases-hl_5',hl_5['biases'])
    # tf.summary.histogram('act-hl_5',l5)
    #
    # tf.summary.histogram('weights-hl_6',hl_6['weights'])
    # tf.summary.histogram('biases-hl_6',hl_6['biases'])
    # tf.summary.histogram('act-hl_6',l6)
    #
    # tf.summary.histogram('weights-hl_7',hl_7['weights'])
    # tf.summary.histogram('biases-hl_7',hl_7['biases'])
    # tf.summary.histogram('act-hl_7',l7)
    #
    # tf.summary.histogram('weights-hl_8',hl_8['weights'])
    # tf.summary.histogram('biases-hl_8',hl_8['biases'])
    # tf.summary.histogram('act-hl_8',l8)
    #
    # tf.summary.histogram('weights-hl_9',hl_9['weights'])
    # tf.summary.histogram('biases-hl_9',hl_9['biases'])
    # tf.summary.histogram('act-hl_9',l9)
    #
    # tf.summary.histogram('weights-hl_10',hl_10['weights'])
    # tf.summary.histogram('biases-hl_10',hl_10['biases'])
    # tf.summary.histogram('act-hl_10',l10)

print "Training"
def train(x):
    tower_grads = []
    opt = tf.train.AdamOptimizer(learning_rate)
    for i in xrange(2):
        with tf.device('/gpu:%d' % i):
            with tf.variable_scope('NN',reuse=i>0):
                prediction = convNeuralNet(x)
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
                tf.summary.scalar('cross_entropy',cost)

                grads = opt.compute_gradients(cost)
                tower_grads.append(grads)
                print grads
                print len(grads)
                #scope.reuse_variables()

        grads = average_gradients(tower_grads)
        apply_gradient_op = opt.apply_gradients(grads)
        train_op = tf.group(apply_gradient_op)

    # cycles of feed forward and backprop

    correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct,'float'))
    tf.summary.scalar('accuracy',accuracy)
    num_epochs = ne

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        saver = tf.train.Saver()
        # UNCOMMENT THIS WHEN RESTARTING FROM Checkpoint
        #saver.restore(sess, tf.train.latest_checkpoint(os.getcwd()+'/models/base/.'))

        sess.run(tf.global_variables_initializer())
        merged_summary = tf.summary.merge_all()

        for epoch in range(num_epochs):
            epoch_loss = 0
            for i in range(int(real_X_9.shape[0])/batch_size):#mnist.train.num_examples/batch_size)): # X.shape[0]
                randidx = np.random.choice(real_X_9.shape[0], batch_size, replace=False)
                epoch_x,epoch_y = real_X_9[randidx,:],real_y_9[randidx,:] #mnist.train.next_batch(batch_size) # X,y
                j,c = sess.run([train_op,cost],feed_dict={x:epoch_x,y:epoch_y,keep_prob:TRAIN_KEEP_PROB})
                if i == 0:
                    [ta] = sess.run([accuracy],feed_dict={x:epoch_x,y:epoch_y,keep_prob:TRAIN_KEEP_PROB})
                    print 'Train Accuracy', ta

                epoch_loss += c
            print '\n','Epoch', epoch + 1, 'completed out of', num_epochs, '\nLoss:',epoch_loss

        #saver.save(sess, os.getcwd()+'/models/base/baseDNN7')
        #saver.export_meta_graph(os.getcwd()+'/models/base/baseDNN7.meta')

        print '\n','Train Accuracy', accuracy.eval(feed_dict={x:real_X_9, y:real_y_9, keep_prob:TRAIN_KEEP_PROB})
        print '\n','Test Accuracy', accuracy.eval(feed_dict={x:test_real_X, y:test_real_y, keep_prob:1.0}) #X, y #mnist.test.images, mnist.test.labels
train(x)

# plt.plot(ta_list)
# plt.show()
# plt.savefig('ta.png')
