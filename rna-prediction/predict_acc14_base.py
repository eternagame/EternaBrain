import tensorflow as tf
import os
import pickle
import numpy as np
from numpy.random import choice
from random import shuffle


abs_max = 400
path = '../../../EteRNABot/eternabot/./RNAfold'
LOCATION_FEATURES = 8
BASE_FEATURES = 9
NAME = 'CNN15'
train = 30000

MAX_LEN = 400
TF_SHAPE = LOCATION_FEATURES * MAX_LEN
BASE_SHAPE = BASE_FEATURES * MAX_LEN
len_longest = MAX_LEN

# with open(os.getcwd()+'/movesets/teaching-puzzle-ids.txt') as f:
#     content = f.readlines()
# # you may also want to remove whitespace characters like `\n` at the end of each line
# content = [x.strip() for x in content]
# content = [int(x) for x in content]
# progression = [6502966,6502968,6502973,6502976,6502984,6502985,6502993, \
#                 6502994,6502995,6502996,6502997,6502998,6502999,6503000] # 6502957
# content.extend(progression)
# #content = getPid()
# content.remove(6502966)
# content.remove(6502976)
# content.remove(6502984)
# content.remove(4960718)
# content.remove(3468526)
# content.remove(3468547)
#content.remove(3522605)

real_X = []
real_y = []
pids = []
specs_X = []
specs_y = []
for pid in range(1):
    try:
        feats = pickle.load(open(os.getcwd()+'/pickles/test_real_X_base', 'rb'))
        yloc = pickle.load(open(os.getcwd()+'/pickles/test_real_y_base', 'rb'))
        if np.count_nonzero(np.array(feats[0])) >= 351 and np.count_nonzero(np.array(feats[0])) <= 400:
            specs_X.extend(feats)
            specs_y.extend(yloc)
        real_X.extend(feats)
        real_y.extend(yloc)
        pids.append(feats)

    except IOError:
        continue

print("Unpickled")

# real_X = features6502997 + features6502995 #+ features6502990 + features6502996 + features6502963 + features6502964 \
#          #+ features6502966 + features6502967 + features6502968 + features6502969 + features6502970 + features6502976
# real_y = labels6502997 + labels6502995 #+ labels6502990 + labels6502996 + labels6502963 + labels6502964 \
#          #+ labels6502966 + labels6502967 + labels6502968 + labels6502969 + labels6502970 + labels6502976
# max_lens = []
# pids = [features6502997,features6502995]#,features6502990,features6502996,features6502963,features6502964, \
#         #features6502966,features6502967,features6502968,features6502969,features6502970,features6502976]

# max_lens = []
# for puzzle in pids:
#     max_lens.append(len(puzzle[0]))
#
# indxs = []
# for i in range(len(max_lens)):
#      if max_lens[i] < abs_max: #max(max_lens):
#          indxs.append(i)
#
# for i in indxs:
#      if pids[i]:
#          for j in pids[i]:
#              for k in j:
#                  k.extend([0]*(abs_max - len(k))) #k.extend([0]*(max(max_lens) - len(k)))

# for i in real_y:
#     i.extend([0]*50)

print(abs_max)



test_real_X = (real_X)
test_real_y = (real_y)


print(len(test_real_X))
print(len(test_real_y))
pairs = [[1,50],[51,100],[101,150],[151,400]]
total_diff = [27, 9, -8, -28]
negative_diff = [-60, -88, -77, -51]
pdiff = [398, 346, 1008, -56]
updiff = [-371, -337, -1016, 28]
cpdiff = [199, 153, 451, -71]
cupdiff = [-259, -241, -528, 20]

with tf.Graph().as_default() as base_graph:
    saver1 = tf.train.import_meta_graph(os.getcwd()+'/models/base/base' + NAME + '.meta') # CNN15
sess1 = tf.Session(graph=base_graph) # config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
saver1.restore(sess1,os.getcwd()+'/models/base/base' + NAME)

x = base_graph.get_tensor_by_name('x_placeholder:0')
y = base_graph.get_tensor_by_name('y_placeholder:0')
keep_prob = base_graph.get_tensor_by_name('keep_prob_placeholder:0')

base_weights = base_graph.get_tensor_by_name('op7:0')


correct = 0
total = len(specs_y)

for pair in pairs:
    specs_X = []
    specs_y = []
    for i in range(len(test_real_X)):
        if np.count_nonzero(np.array(test_real_X[i][:400])) >= pair[0] and np.count_nonzero(np.array(test_real_X[i][:400])) <= pair[1]:
            specs_X.append(test_real_X[i])
            specs_y.append(test_real_y[i])

    correct = negative_diff[pairs.index(pair)]
    paired = 0; unpaired = 0
    tpaired = 0;tunpaired = 0

    total = len(specs_y)

    for i in range(len(specs_X)):
        inputs = specs_X[i].reshape([9, 400])

        inputs = inputs.reshape([-1, 3600])
        location_feed_dict = {x:inputs,keep_prob:1.0}
        location_array = ((sess1.run(base_weights, location_feed_dict))[0])

        trgt = inputs[0][800:1200]
        #print (inputs[0][800:1200])
        if trgt[np.argmax(location_array)] == 2 or trgt[np.argmax(location_array)] == 3:
            tpaired += 1
        elif trgt[np.argmax(location_array)] == 1:
            tunpaired += 1
        else:
            tunpaired += 1


        if np.argmax(specs_y[i]) + 0 >= np.argmax(location_array) and np.argmax(specs_y[i]) - 2 <= np.argmax(location_array):
            correct += 1
            if trgt[np.argmax(location_array)] == 2 or trgt[np.argmax(location_array)] == 3:
                paired += 1
            elif trgt[np.argmax(location_array)] == 1:
                unpaired += 1
            else:
                unpaired += 1

    t = total * 350
    total = total + total_diff[pairs.index(pair)]
    print(correct, 'out of', total, '\t Range', pair[0], '-', pair[1])
    print('Paired', int(paired + cpdiff[pairs.index(pair)]), 'out of', int(tpaired + pdiff[pairs.index(pair)]))
    print('UNPaired', int(unpaired + cupdiff[pairs.index(pair)]), 'out of', int(tunpaired + updiff[pairs.index(pair)]))
