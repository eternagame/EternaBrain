import numpy as np
import os
import tensorflow as tf
import pickle
from tf_funcs import average_gradients

# enc0 = np.array([[[[1,2,3,4],[0,1,0,1],[-33,0,0,0]],[[1,2,3,4],[0,1,1,0],[-23,0,0,0]]],[[[3,3,3,3],[0,0,0,0],[2,0,0,0]],[[1,1,1,0],[1,0,1,0],[-23,0,0,0]]]])
# ms0 = np.array([[[2,1],[4,3]],[[1,6],[2,9]]])
# enc = np.array([[[1,2,3,4],[0,1,0,1],[1,1,1,1],[-3,0,0,0]],[[4,3,2,1],[1,0,1,0],[0,0,0,0],[9,0,0,0]]])
# out = np.array([[4,2],[3,3]])

features6502997 = pickle.load(open(os.getcwd()+'/pickles/X-6502997-dev','rb'))
labels6502997 = pickle.load(open(os.getcwd()+'/pickles/y-6502997-loc-dev','rb'))
print 'unpickled 1'
features6502995 = pickle.load(open(os.getcwd()+'/pickles/X-6502995-dev','rb'))
labels6502995 = pickle.load(open(os.getcwd()+'/pickles/y-6502995-loc-dev','rb'))

features6502990 = pickle.load(open(os.getcwd()+'/pickles/X-6502990-dev','rb'))
labels6502990 = pickle.load(open(os.getcwd()+'/pickles/y-6502990-loc-dev','rb'))

features6502996 = pickle.load(open(os.getcwd()+'/pickles/X-6502996-dev','rb'))
labels6502996 = pickle.load(open(os.getcwd()+'/pickles/y-6502996-loc-dev','rb'))

features6502963 = pickle.load(open(os.getcwd()+'/pickles/X-6502963','rb'))
labels6502963 = pickle.load(open(os.getcwd()+'/pickles/y-6502963-loc','rb'))

features6502964 = pickle.load(open(os.getcwd()+'/pickles/X-6502964','rb'))
labels6502964 = pickle.load(open(os.getcwd()+'/pickles/y-6502964-loc','rb'))

features6502966 = pickle.load(open(os.getcwd()+'/pickles/X-6502966','rb'))
labels6502966 = pickle.load(open(os.getcwd()+'/pickles/y-6502966-loc','rb'))

features6502967 = pickle.load(open(os.getcwd()+'/pickles/X-6502967','rb'))
labels6502967 = pickle.load(open(os.getcwd()+'/pickles/y-6502967-loc','rb'))

features6502968 = pickle.load(open(os.getcwd()+'/pickles/X-6502968','rb'))
labels6502968 = pickle.load(open(os.getcwd()+'/pickles/y-6502968-loc','rb'))

features6502969 = pickle.load(open(os.getcwd()+'/pickles/X-6502969','rb'))
labels6502969 = pickle.load(open(os.getcwd()+'/pickles/y-6502969-loc','rb'))

features6502970 = pickle.load(open(os.getcwd()+'/pickles/X-6502970','rb'))
labels6502970 = pickle.load(open(os.getcwd()+'/pickles/y-6502970-loc','rb'))

features6502976 = pickle.load(open(os.getcwd()+'/pickles/X-6502976','rb'))
labels6502976 = pickle.load(open(os.getcwd()+'/pickles/y-6502976-loc','rb'))

print "Unpickled 2"

real_X = features6502997 + features6502995 + features6502990 + features6502996 + features6502963+features6502964 \
         + features6502966 + features6502967 + features6502968 + features6502969 + features6502970 + features6502976
real_y = labels6502997 + labels6502995 + labels6502990 + labels6502996 + labels6502963 + labels6502964 \
         + labels6502966 + labels6502967 + labels6502968 + labels6502969 + labels6502970 + labels6502976
max_lens = []
max_labels = []
pids = [features6502997,features6502995,features6502990,features6502996,features6502963,features6502964, \
        features6502966,features6502967,features6502968,features6502969,features6502970,features6502976]
locations = [labels6502997,labels6502995,labels6502990,labels6502996,labels6502963,labels6502964, \
                labels6502966,labels6502967,labels6502968,labels6502969,labels6502970,labels6502976]
for puzzle,loc in zip(pids,locations):
    max_lens.append(len(puzzle[0][0]))
    max_labels.append(len(loc[0]))

indxs = []
indxs_locations = []
# for i,j in zip(max_lens, max_labels):
#     if i < max(max_lens):
#         indxs.append(max_lens.index(i))
#     if j < max(max_labels):
#         indxs_locations.append(max_labels.index(i))

# for i in range(len(max_labels)):
#     if max_lens[i] < max(max_lens):
#         indxs.append(i)
#     elif max_labels[i] < max(max_labels):
#         indxs_locations.append(i)

for i in range(len(max_lens)):
    if max_lens[i] < max(max_lens):
        indxs.append(i)

for i in range(len(max_labels)):
    if max_labels[i] < max(max_labels):
        indxs_locations.append(i)

for i in indxs:
     if pids[i]:
         for j in pids[i]:
             for k in j:
                 k.extend([0]*(max(max_lens) - len(k)))

for i in indxs_locations:
    if locations[i]:
        for j in locations[i]:
            j.extend([0]*(max(max_labels) - len(j)))


TRAIN_KEEP_PROB = 1.0
TEST_KEEP_PROB = 1.0
learning_rate = 0.0001
ne = 300
#tb_path = '/tensorboard/baseDNN-500-10-10-50-100'

train = 1800000
test = 20
num_nodes = 250
len_puzzle = max(max_lens)

TF_SHAPE = 6 * len_puzzle

ta_list = []

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

n_classes = max(max_labels)
batch_size = 128 # load 100 features at a time

x = tf.placeholder('float',[None,TF_SHAPE],name="x_placeholder") # 216 with enc0
y = tf.placeholder('float',name='y_placeholder')
keep_prob = tf.placeholder('float',name="keep_prob_placeholder")

# enc = enc0.reshape([-1,16])
# ms = ms0#.reshape([-1,4])
#
# test_enc = test_enc0.reshape([-1,16])
# test_ms = test_ms0

#e1 = tf.reshape(enc0,[])

def neuralNet(data):
    hl_1 = {'weights':tf.get_variable('Weights1',[TF_SHAPE,n_nodes_hl1],initializer=tf.random_normal_initializer()),
            'biases':tf.get_variable('Biases1',[n_nodes_hl1],initializer=tf.random_normal_initializer())}

    hl_2 = {'weights':tf.get_variable('Weights2',[n_nodes_hl1, n_nodes_hl2],initializer=tf.random_normal_initializer()),
            'biases':tf.get_variable('Biases2',[n_nodes_hl2],initializer=tf.random_normal_initializer())}

    hl_3 = {'weights':tf.get_variable('Weights3',[n_nodes_hl2, n_nodes_hl3],initializer=tf.random_normal_initializer()),
            'biases':tf.get_variable('Biases3',[n_nodes_hl3],initializer=tf.random_normal_initializer())}

    hl_4 = {'weights':tf.get_variable('Weights4',[n_nodes_hl3, n_nodes_hl4],initializer=tf.random_normal_initializer()),
            'biases':tf.get_variable('Biases4',[n_nodes_hl4],initializer=tf.random_normal_initializer())}

    hl_5 = {'weights':tf.get_variable('Weights5',[n_nodes_hl4, n_nodes_hl5],initializer=tf.random_normal_initializer()),
            'biases':tf.get_variable('Biases5',[n_nodes_hl5],initializer=tf.random_normal_initializer())}

    # hl_6 = {'weights':tf.get_variable(initializer=tf.random_normal_initializer(),([n_nodes_hl5, n_nodes_hl6]),'Weights'),
    #         'biases':tf.get_variable(initializer=tf.random_normal_initializer(),([n_nodes_hl6]),name='Biases')}
    #
    # hl_7 = {'weights':tf.get_variable(initializer=tf.random_normal_initializer(),([n_nodes_hl6, n_nodes_hl7]),name='Weights'),
    #         'biases':tf.get_variable(initializer=tf.random_normal_initializer(),([n_nodes_hl7]),name='Biases')}
    #
    # hl_8 = {'weights':tf.get_variable(initializer=tf.random_normal_initializer(),([n_nodes_hl7, n_nodes_hl8]),name='Weights'),
    #         'biases':tf.get_variable(initializer=tf.random_normal_initializer(),([n_nodes_hl8]),name='Biases')}
    #
    # hl_9 = {'weights':tf.get_variable(initializer=tf.random_normal_initializer(),([n_nodes_hl8, n_nodes_hl9]),name='Weights'),
    #         'biases':tf.get_variable(initializer=tf.random_normal_initializer(),([n_nodes_hl9]),name='Biases')}
    #
    # hl_10 = {'weights':tf.get_variable(initializer=tf.random_normal_initializer(),([n_nodes_hl9, n_nodes_hl10]),name='Weights'),
    #         'biases':tf.get_variable(initializer=tf.random_normal_initializer(),([n_nodes_hl10]),name='Biases')}

    output_layer = {'weights':tf.get_variable('Weights-outputlayer',[n_nodes_hl5, n_classes],initializer=tf.random_normal_initializer()),
            'biases':tf.get_variable('Biases-outputlayer',[n_classes],initializer=tf.random_normal_initializer())}

    # l1 = tf.add(tf.matmul(data, hl_1['weights']), hl_1['biases'])
    # l1 = tf.nn.sigmoid(l1, name='op1')
    l1 = tf.add(tf.matmul(data, hl_1['weights']), hl_1['biases'])
    l1 = tf.nn.sigmoid(l1, name='op1')

    l2 = tf.add(tf.matmul(l1, hl_2['weights']), hl_2['biases'])
    l2 = tf.nn.sigmoid(l2, name='op2')

    l3 = tf.add(tf.matmul(l2, hl_3['weights']), hl_3['biases'])
    l3 = tf.nn.sigmoid(l3, name='op3')

    l4 = tf.add(tf.matmul(l3, hl_4['weights']), hl_4['biases'])
    l4 = tf.nn.sigmoid(l4, name='op4')

    l5 = tf.add(tf.matmul(l4, hl_5['weights']), hl_5['biases'])
    l5 = tf.nn.sigmoid(l5, name='op5')

    # l6 = tf.add(tf.matmul(l5, hl_6['weights']), hl_6['biases'])
    # l6 = tf.nn.relu(l6)
    #
    # l7 = tf.add(tf.matmul(l6, hl_7['weights']), hl_7['biases'])
    # l7 = tf.nn.relu(l7)
    #
    # l8 = tf.add(tf.matmul(l7, hl_8['weights']), hl_8['biases'])
    # l8 = tf.nn.relu(l8)
    #
    # l9 = tf.add(tf.matmul(l8, hl_9['weights']), hl_9['biases'])
    # l9 = tf.nn.relu(l9)
    #
    # l10 = tf.add(tf.matmul(l9, hl_10['weights']), hl_10['biases'])
    # l10 = tf.nn.relu(l10)

    dropout = tf.nn.dropout(l5,keep_prob, name='op6')
    ol = tf.add(tf.matmul(dropout, output_layer['weights']), output_layer['biases'], name='op7')

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

    return ol

print "Training"

def train(x):
    tower_grads = []
    opt = tf.train.AdamOptimizer(learning_rate)
    for i in xrange(8):
        with tf.device('/gpu:%d' % i):
            with tf.variable_scope('NN',reuse=i>0):
                prediction = neuralNet(x)
                #with tf.name_scope('cross_entropy'):
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
                tf.summary.scalar('cross_entropy',cost)

                #with tf.name_scope('train'):
                #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost) # learning rate = 0.001

                #with tf.name_scope('accuracy'):

                grads = opt.compute_gradients(cost)
                tower_grads.append(grads)
                print grads
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
        #saver.restore(sess, tf.train.latest_checkpoint(os.getcwd()+'/models/location/.'))

        sess.run(tf.global_variables_initializer())
        merged_summary = tf.summary.merge_all()
        #writer = tf.summary.FileWriter(os.getcwd()+tb_path)
        #writer.add_graph(sess.graph)

        for epoch in range(num_epochs):
            epoch_loss = 0
            for i in range(int(real_X_9.shape[0])/batch_size):#mnist.train.num_examples/batch_size)): # X.shape[0]
                randidx = np.random.choice(real_X_9.shape[0], batch_size, replace=False)
                epoch_x,epoch_y = real_X_9[randidx,:],real_y_9[randidx,:] #mnist.train.next_batch(batch_size) # X,y
                j,c = sess.run([train_op,cost],feed_dict={x:epoch_x,y:epoch_y,keep_prob:TRAIN_KEEP_PROB})
                if i == 0:
                    [ta] = sess.run([accuracy],feed_dict={x:epoch_x,y:epoch_y,keep_prob:TRAIN_KEEP_PROB})
                    print 'Train Accuracy', ta
                if epoch % 50 == 0 and i == 0:
                    saver.save(sess,os.getcwd()+'/models/location/locationDNN4.ckpt')
                    print 'Checkpoint saved'
                    # ta_list.append(ta)
                # if i % 5 == 0:
                #     s = sess.run(merged_summary,feed_dict={x:epoch_x,y:epoch_y,keep_prob:TRAIN_KEEP_PROB})
                #     writer.add_summary(s,i)

                epoch_loss += c
            print '\n','Epoch', epoch + 1, 'completed out of', num_epochs, '\nLoss:',epoch_loss

        saver.save(sess, os.getcwd()+'/models/location/locationDNN4')
        saver.export_meta_graph(os.getcwd()+'/models/location/locationDNN4.meta')

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
        saver.save(sess,os.getcwd()+'/models/location/baseDNN')
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

# plt.plot(ta_list)
# plt.show()
# plt.savefig('ta.png')
