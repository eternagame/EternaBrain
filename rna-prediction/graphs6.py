# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 19:30:09 2017

@author: rohankoodli
"""

import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn; seaborn.set()
import pickle

filepath = os.getcwd() + '/movesets/move-set-11-14-2016.txt'

#puzzle = raw_input("Puzzle ID: ")
plt.figure(1)

y_gmm_6892346 = pickle.load(open(os.getcwd() + '/pickles/gmm-6892346','rb'))
transf_6892346 = pickle.load(open(os.getcwd() + '/pickles/pca-6892346','rb'))
pc_6892346 = pickle.load(open(os.getcwd() + '/pickles/components-6892346','rb'))

#data, users = read_movesets_pid(filepath,7254760)
plt.subplot(211)
plt.scatter(transf_6892346[:,0], transf_6892346[:,1],c=y_gmm_6892346, cmap='RdYlBu',s=150)
plt.suptitle("Puzzle " + puzzle,fontsize=18)
plt.xlabel('Component 1 (Explained Variance: %s)'%(pc_6892346[0]),fontsize=14)
plt.ylabel('Component 2 (Explained Variance: %s)'%(pc_6892346[1]),fontsize=14)

y_gmm_6892348 = pickle.load(open(os.getcwd() + '/pickles/gmm-6892348','rb'))
transf_6892348 = pickle.load(open(os.getcwd() + '/pickles/pca-6892348','rb'))
pc_6892348 = pickle.load(open(os.getcwd() + '/pickles/components-6892348','rb'))

#data, users = read_movesets_pid(filepath,7254760)
plt.subplot(212)
plt.scatter(transf_6892346[:,0], transf_6892346[:,1],c=y_gmm_6892346, cmap='RdYlBu',s=150)
plt.suptitle("Puzzle " + puzzle,fontsize=18)
plt.xlabel('Component 1 (Explained Variance: %s)'%(pc_6892347[0]),fontsize=14)
plt.ylabel('Component 2 (Explained Variance: %s)'%(pc_6892347[1]),fontsize=14)

y_gmm_6892348 = pickle.load(open(os.getcwd() + '/pickles/gmm-6892348','rb'))
transf_6892348 = pickle.load(open(os.getcwd() + '/pickles/pca-6892348','rb'))
pc_6892348 = pickle.load(open(os.getcwd() + '/pickles/components-6892348','rb'))

#data, users = read_movesets_pid(filepath,7254760)
plt.subplot(213)
plt.scatter(transf_6892348[:,0], transf_6892348[:,1],c=y_gmm_6892348, cmap='RdYlBu',s=150)
plt.suptitle("Puzzle " + puzzle,fontsize=18)
plt.xlabel('Component 1 (Explained Variance: %s)'%(pc_6892348[0]),fontsize=14)
plt.ylabel('Component 2 (Explained Variance: %s)'%(pc_6892348[1]),fontsize=14)

