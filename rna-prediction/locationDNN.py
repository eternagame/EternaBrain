import numpy as np
import os
from readData import read_movesets_pid, read_structure
from encodeRNA import encode_movesets, encode_structure
import tensorflow as tf
import pickle
from sklearn.cross_validation import train_test_split

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/Users/rohankoodli/Documents/MNIST/",one_hot=True)

# enc0 = np.array([[[[1,2,3,4],[0,1,0,1],[-33,0,0,0]],[[1,2,3,4],[0,1,1,0],[-23,0,0,0]]],[[[3,3,3,3],[0,0,0,0],[2,0,0,0]],[[1,1,1,0],[1,0,1,0],[-23,0,0,0]]]])
# ms0 = np.array([[[2,1],[4,3]],[[1,6],[2,9]]])
# enc = np.array([[[1,2,3,4],[0,1,0,1],[1,1,1,1],[-3,0,0,0]],[[4,3,2,1],[1,0,1,0],[0,0,0,0],[9,0,0,0]]])
# out = np.array([[4,2],[3,3]])


real_X = pickle.load(open(os.getcwd()+'/pickles/X-6892348','rb'))
real_y = pickle.load(open(os.getcwd()+'/pickles/y-6892348-loc','rb'))
'''
for i in (real_X[432:]):
    current_structure = i[1]
    energy = i[3]
    current_structure.insert(len(current_structure),0)
    energy.insert(len(energy),0.0)
'''

train = 1000
test = 20

real_X_9 = np.array(real_X[0:train]).reshape([-1,340])
real_y_9 = np.array(real_y[0:train])
test_real_X = np.array(real_X[train:train+test]).reshape([-1,340])
test_real_y = np.array(real_y[train:train+test])

#real_X_9, test_real_X, real_y_9, test_real_y = np.array(train_test_split(real_X[0:500],real_y[0:500],test_size=0.2))
#real_X_9, test_real_X, real_y_9, test_real_y = np.array(real_X_9).reshape([-1,340]), np.array(test_real_X).reshape([-1,340]), np.array(real_y_9), np.array(test_real_y)

# enc0 = np.array([[[1,2,3,4],[0,1,0,1],[-33,0,0,0],[1,1,1,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]]])
# ms0 = np.array([[1,6],[2,7],[2,7],[2,7],[2,7],[2,7],[2,7],[2,7],[2,7]])
# ms0 = np.array([[1,0,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0]]) # just base
#
# test_enc0 = np.array([[[2,3,3,2],[0,0,0,0],[6,0,0,0],[0,0,1,1]],[[1,2,3,4],[0,1,0,1],[-33,0,0,0],[1,1,1,1]]])
# test_ms0 = np.array([[4,20],[3,15]])
# test_ms0 = np.array([[0,0,0,1],[1,0,0,0]]) # just base

n_nodes_hl1 = 700 # hidden layer 1
n_nodes_hl2 = 700
n_nodes_hl3 = 700
n_nodes_hl4 = 700
n_nodes_hl5 = 700
n_nodes_hl6 = 700
n_nodes_hl7 = 700
n_nodes_hl8 = 700
n_nodes_hl9 = 700
n_nodes_hl10 = 700

n_classes = 121
batch_size = 100 # load 100 features at a time


x = tf.placeholder('float',[None,340]) # 16 with enc0
y = tf.placeholder('float')

enc = enc0.reshape([-1,16])
ms = ms0#.reshape([-1,4])

test_enc = test_enc0.reshape([-1,16])
test_ms = test_ms0

#e1 = tf.reshape(enc0,[])

def neuralNet(data):
    hl_1 = {'weights':tf.Variable(tf.random_normal([340, n_nodes_hl1])),
            'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hl_2 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
            'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hl_3 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
            'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    hl_4 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
            'biases':tf.Variable(tf.random_normal([n_nodes_hl4]))}

    hl_5 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl4, n_nodes_hl5])),
            'biases':tf.Variable(tf.random_normal([n_nodes_hl5]))}

    hl_6 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl5, n_nodes_hl6])),
            'biases':tf.Variable(tf.random_normal([n_nodes_hl6]))}

    hl_7 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl6, n_nodes_hl7])),
            'biases':tf.Variable(tf.random_normal([n_nodes_hl7]))}

    hl_8 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl7, n_nodes_hl8])),
            'biases':tf.Variable(tf.random_normal([n_nodes_hl8]))}

    hl_9 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl8, n_nodes_hl9])),
            'biases':tf.Variable(tf.random_normal([n_nodes_hl9]))}

    hl_10 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl9, n_nodes_hl10])),
            'biases':tf.Variable(tf.random_normal([n_nodes_hl10]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl10, n_classes])),
            'biases':tf.Variable(tf.random_normal([n_classes]))}

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

    ol = tf.matmul(l10, output_layer['weights']) + output_layer['biases']

    return ol


def train(x):
    prediction = neuralNet(x)
    #print prediction
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost) # learning rate = 0.001

    # cycles of feed forward and backprop
    num_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            epoch_loss = 0
            for _ in range(int(real_X_9.shape[0])):#mnist.train.num_examples/batch_size)): # X.shape[0]
                epoch_x,epoch_y = real_X_9,real_y_9 #mnist.train.next_batch(batch_size) # X,y
                _,c = sess.run([optimizer,cost],feed_dict={x:epoch_x,y:epoch_y})
                epoch_loss += c
            print 'Epoch', epoch + 1, 'completed out of', num_epochs, '\nLoss:',epoch_loss,'\n'

        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))

        print 'Accuracy', accuracy.eval(feed_dict={x:test_real_X, y:test_real_y}) #X, y #mnist.test.images, mnist.test.labels


train(x)
