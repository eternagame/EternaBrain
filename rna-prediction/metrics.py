
import numpy as np
import os
import tensorflow as tf
import pickle

train = 30000
test = 100
len_puzzle = 400
TF_SHAPE = 9 * len_puzzle

with open(os.getcwd()+'/movesets/teaching-puzzle-ids.txt') as f:
    content = f.readlines()
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
    try:
        feats = pickle.load(open(os.getcwd()+'/pickles/X5-exp-loc-'+str(pid),'rb'))
        ybase = pickle.load(open(os.getcwd()+'/pickles/y5-exp-base-'+str(pid),'rb'))
        yloc = pickle.load(open(os.getcwd()+'/pickles/y5-exp-loc-'+str(pid),'rb'))
        for i in range(len(feats)):
            feats[i].append(yloc[i])
        real_X.extend(feats)
        real_y.extend(ybase)
        pids.append(feats)
    except IOError:
        continue

max_lens = []
for puzzle in pids:
    max_lens.append(len(puzzle[0][0]))

abs_max = 400
indxs = []
for i in range(len(max_lens)):
     if max_lens[i] < abs_max: #max(max_lens):
         indxs.append(i)

for i in indxs:
     if pids[i]:
         for j in pids[i]:
             for k in j:
                 k.extend([0]*(abs_max - len(k))) #k.extend([0]*(max(max_lens) - len(k)))

print abs_max

testtest = np.array(real_X[train:train+test]).reshape([-1,TF_SHAPE])

real_X_9 = np.array(real_X[0:train]).reshape([-1,TF_SHAPE])
real_y_9 = np.array(real_y[0:train])
test_real_X = np.array(real_X[train:train+test]).reshape([-1,TF_SHAPE])
test_real_y = np.array(real_y[train:train+test])

print len(real_X), len(real_y)
print np.array(real_X).shape, np.array(real_y).shape

with tf.Graph().as_default() as base_graph:
    saver1 = tf.train.import_meta_graph(os.getcwd()+'/models/base/baseCNN15.meta') # CNN15
sess1 = tf.Session(graph=base_graph) # config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
saver1.restore(sess1,os.getcwd()+'/models/base/baseCNN15')

x = base_graph.get_tensor_by_name('x_placeholder:0')
y = base_graph.get_tensor_by_name('y_placeholder:0')
keep_prob = base_graph.get_tensor_by_name('keep_prob_placeholder:0')

base_weights = base_graph.get_tensor_by_name('op7:0')

base_feed_dict={x:test_real_X[0:1],keep_prob:1.0}
base_array = ((sess1.run(base_weights,base_feed_dict))[0])
print base_array
#print 'Prediction',sess1.run(tf.argmax(prediction,1), feed_dict={x:testtest, keep_prob:1})

base_change = np.argmax(base_array)
print base_change
print np.argmax(test_real_y[0])
#true_y = np
prediction, truth = [],[]
for i in range(50):
    base_feed_dict={x:test_real_X[i:i+1],keep_prob:1.0}
    base_array = ((sess1.run(base_weights,base_feed_dict))[0])
    base_change = np.argmax(base_array)
    real_base = np.argmax(test_real_y[0])
    prediction.append(base_change)
    truth.append(real_base)
    print "Test %i complete" % (i+1)

confusion = tf.confusion_matrix(labels=truth,predictions=prediction,num_classes=4)
sess = tf.Session()
with sess.as_default():
    print confusion.eval()
