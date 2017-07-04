import tensorflow as tf
import os
import pickle

testtest = pickle.load(open(os.getcwd()+'/pickles/testtest','rb'))

TF_SHAPE = 480

sess = tf.Session()
saver = tf.train.import_meta_graph(os.getcwd()+'/models/locationDNN.meta')
saver.restore(sess,os.getcwd()+'/models/locationDNN')

graph = tf.get_default_graph()

x = graph.get_tensor_by_name('x_placeholder:0')
y = graph.get_tensor_by_name('y_placeholder:0')
keep_prob = graph.get_tensor_by_name('keep_prob_placeholder:0')

feed_dict={x:testtest,keep_prob:1.0}
op7 = graph.get_tensor_by_name('op7:0')

print sess.run(op7,feed_dict)
