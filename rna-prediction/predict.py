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

with tf.Graph().as_default() as base_graph:
  saver1 = tf.train.import_meta_graph(os.getcwd()+'/models/baseDNN3.meta')
sess1 = tf.Session(graph=base_graph)
saver1.restore(sess1,os.getcwd()+'/models/baseDNN3')

x = base_graph.get_tensor_by_name('x_placeholder:0')
y = base_graph.get_tensor_by_name('y_placeholder:0')
keep_prob = base_graph.get_tensor_by_name('keep_prob_placeholder:0')

base_weights = base_graph.get_tensor_by_name('op7:0')

base_feed_dict={x:inputs,keep_prob:1.0}

with tf.Graph().as_default() as location_graph:
  saver2 = tf.train.import_meta_graph(os.getcwd()+'/models/locationDNN.meta')
sess2 = tf.Session(graph=location_graph)
saver2.restore(sess2,os.getcwd()+'/models/locationDNN')

x2 = location_graph.get_tensor_by_name('x_placeholder:0')
y2 = location_graph.get_tensor_by_name('y_placeholder:0')
keep_prob2 = location_graph.get_tensor_by_name('keep_prob_placeholder:0')

location_weights = location_graph.get_tensor_by_name('op7:0')

location_feed_dict = {x2:inputs,keep_prob2:1.0}
