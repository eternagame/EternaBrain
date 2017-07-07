import tensorflow as tf
import os
import pickle
import numpy as np

base_seq = [1,1,1,1,2,1,1,1,3,3,1,3,2] + ([0]*95)
current_struc = [1,1,1,1,1,1,1,1,1,1,1,1,1] + ([0]*95)
target_struc = [2,1,1,2,2,1,1,1,3,3,1,1,3] + ([0]*95)
current_energy = [0.0] + ([0]*107)
target_energy = [7.9] + ([0]*107)
locks = [1,1,1,1,1,2,2,2,1,1,1,1,1] + ([0]*95)

TF_SHAPE = 648

inputs = np.array([base_seq,current_struc,target_struc,current_energy,target_energy,locks])
inputs = inputs.reshape([-1,TF_SHAPE])

sess = tf.Session()
saver = tf.train.import_meta_graph(os.getcwd()+'/models/baseDNN3.meta')
saver.restore(sess,os.getcwd()+'/models/baseDNN3')

graph = tf.get_default_graph()

x = graph.get_tensor_by_name('x_placeholder:0')
y = graph.get_tensor_by_name('y_placeholder:0')
keep_prob = graph.get_tensor_by_name('keep_prob_placeholder:0')

feed_dict={x:inputs,keep_prob:1.0}
op7 = graph.get_tensor_by_name('op7:0')

print sess.run((op7),feed_dict)
print np.argmax((sess.run(op7,feed_dict))[0])
print sess.run(tf.argmax(op7),feed_dict)
