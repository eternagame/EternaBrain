# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 18:09:56 2017

@author: rohankoodli
"""

import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn; seaborn.set()
import pickle
from readData import read_movesets_pid

filepath = os.getcwd() + '/movesets/move-set-11-14-2016.txt'

puzzle = raw_input("Puzzle ID: ")

y_gmm = pickle.load(open(os.getcwd() + '/pickles/gmm-' + puzzle,'rb'))
transf = pickle.load(open(os.getcwd() + '/pickles/pca-' + puzzle,'rb'))
#data, users = read_movesets_pid(filepath,7254760)

plt.scatter(transf[:,0], transf[:,1],c=y_gmm, cmap='RdYlBu',s=150)
plt.suptitle("Puzzle " + puzzle,fontsize=18)

'''
c1,c2,c3,c4,c5,c6,c7 = [],[],[],[],[],[],[]
clusters = [c1,c2,c3,c4,c5,c6,c7]
for i in range(len(users)):
  if y_gmm[i] == 0:
    c1.append(users[i])
  elif y_gmm[i] == 1:
    c2.append(users[i])
  elif y_gmm[i] == 2:
    c3.append(users[i])
  elif y_gmm[i] == 3:
    c4.append(users[i])
  elif y_gmm[i] == 4:
    c5.append(users[i])
  elif y_gmm[i] == 5:
    c6.append(users[i]) 
  elif y_gmm[i] == 6:
    c7.append(users[i])

for i in range(len(clusters)):
  print '\nCluster %i:' %(i+1),np.count_nonzero(y_gmm == i)
  print clusters[i]
'''