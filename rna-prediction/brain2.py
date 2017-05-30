import tensorflow as tf
import numpy as np

enc0 = np.array([[[1,2,3,4],[0,1,0,1],[-33,0,0,0],[1,1,1,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]],[[2,3,3,2],[0,0,0,0],[9,0,0,0],[0,0,0,1]]],dtype=np.float32)
ms0 = np.array([[1,6],[2,7],[2,7]],dtype=np.float32)

a = enc0#.reshape([-1,16])
k = ms0
flip = [slice(None, None, -1), slice(None, None, -1)]
k = k[flip]

a=a.astype(np.float32)
a_tensor = tf.reshape(a, [1, 2, 16, 1])
k_weight = tf.reshape((k), [2,2,1,1])
print a_tensor
print k_weight
c=tf.nn.conv2d(a_tensor, k_weight,padding='VALID',strides=[1, 1, 1, 1])
sess=tf.Session()
print (c.eval(session=sess))
