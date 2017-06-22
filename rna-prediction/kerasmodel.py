from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
import keras
import os
import pickle
import numpy as np
import tensorflow as tf

TRAIN_KEEP_PROB = 1.0
TEST_KEEP_PROB = 1.0
learning_rate = 0.00001
#tb_path = '/tensorboard/baseDNN-500-10-10-50-100'

train = 5000
test = 1
num_nodes = 250
len_puzzle = 38

TF_SHAPE = 4 * len_puzzle

real_X = pickle.load(open(os.getcwd()+'/pickles/X-6502994','rb'))
real_y = pickle.load(open(os.getcwd()+'/pickles/y-6502994','rb'))

#testtest = np.array(real_X[train:train+test]).reshape([-1,TF_SHAPE])

X = np.array(real_X[0:train]).reshape([-1,TF_SHAPE])
y = np.array(real_y[0:train])
test_X = np.array(real_X[train:train+test]).reshape([-1,TF_SHAPE])
test_y = np.array(real_y[train:train+test])


model = Sequential()

model.add(Dense(units=50, input_dim=152))
model.add(Activation('relu'))

model.add(Dense(units=10))
model.add(Activation('softmax'))

# model.add(Dense(units=50))
# model.add(Activation('relu'))
#
# model.add(Dense(units=50))
# model.add(Activation('relu'))
#
# model.add(Dense(units=50))
# model.add(Activation('relu'))
#
# model.add(Dense(units=50))
# model.add(Activation('relu'))
#
# model.add(Dense(units=50))
# model.add(Activation('relu'))
#
# model.add(Dense(units=50))
# model.add(Activation('relu'))
#
# model.add(Dense(units=50))
# model.add(Activation('relu'))
#
# model.add(Dense(units=50))
# model.add(Activation('relu'))

model.add(Dropout(TRAIN_KEEP_PROB))

model.add(Dense(units=4))
model.add(Activation('relu'))

adam = keras.optimizers.Adam(lr=learning_rate, clipnorm=1)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=adam,
              metrics=['accuracy'])

model.fit(X, y, epochs=500, batch_size=500)

loss_and_metrics = model.evaluate(test_X, test_y, batch_size=128)
#classes = model.predict(X, batch_size=128)
