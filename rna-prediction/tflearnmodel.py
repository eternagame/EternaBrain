import numpy as np
import os
import tensorflow as tf
import pickle
from sklearn.cross_validation import train_test_split
import tflearn
#from matplotlib import pyplot as plt

# enc0 = np.array([[[[1,2,3,4],[0,1,0,1],[-33,0,0,0]],[[1,2,3,4],[0,1,1,0],[-23,0,0,0]]],[[[3,3,3,3],[0,0,0,0],[2,0,0,0]],[[1,1,1,0],[1,0,1,0],[-23,0,0,0]]]])
# ms0 = np.array([[[2,1],[4,3]],[[1,6],[2,9]]])
# enc = np.array([[[1,2,3,4],[0,1,0,1],[1,1,1,1],[-3,0,0,0]],[[4,3,2,1],[1,0,1,0],[0,0,0,0],[9,0,0,0]]])
# out = np.array([[4,2],[3,3]])

# features6502997 = pickle.load(open(os.getcwd()+'/pickles/X-6502997','rb'))
# labels6502997 = pickle.load(open(os.getcwd()+'/pickles/y-6502997','rb'))
# features6502998 = pickle.load(open(os.getcwd()+'/pickles/X-6502998','rb'))
# labels6502998 = pickle.load(open(os.getcwd()+'/pickles/y-6502998','rb'))

real_X = pickle.load(open(os.getcwd()+'/pickles/X-6502994','rb'))
real_y = pickle.load(open(os.getcwd()+'/pickles/y-6502994','rb'))


TRAIN_KEEP_PROB = 1.0
TEST_KEEP_PROB = 1.0
learning_rate = 0.00001
ne = 100
#tb_path = '/tensorboard/baseDNN-500-10-10-50-100'

train = 2000
test = 120
num_nodes = 250
len_puzzle = 38

TF_SHAPE = 4 * len_puzzle

ta_list = []

#testtest = np.array(real_X[train:train+test]).reshape([-1,TF_SHAPE])

# real_X_9 = np.array(real_X[0:train]).reshape([-1,TF_SHAPE])
# real_y_9 = np.array(real_y[0:train])
# test_real_X = np.array(real_X[train:train+test]).reshape([-1,TF_SHAPE])
# test_real_y = np.array(real_y[train:train+test])

real_X_9, test_real_X, real_y_9, test_real_y = np.array(train_test_split(real_X[0:train],real_y[0:train],test_size=0.001))
real_X_9, test_real_X, real_y_9, test_real_y = np.array(real_X_9).reshape([-1,TF_SHAPE]), np.array(test_real_X).reshape([-1,TF_SHAPE]), np.array(real_y_9), np.array(test_real_y)

tflearn.init_graph(num_cores=1)

net = tflearn.input_data(shape=[None, TF_SHAPE])
net = tflearn.fully_connected(net, num_nodes, activation='relu')
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 4, activation='relu')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy', metric='accuracy')

model = tflearn.DNN(net)
model.fit(real_X_9,real_y_9)
