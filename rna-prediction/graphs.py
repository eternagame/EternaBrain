# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 18:09:56 2017

@author: rohankoodli
"""

from sklearn.mixture import GaussianMixture
import os
from matplotlib import pyplot as plt
import seaborn; seaborn.set()
import pickle

puzzle = raw_input("Puzzle ID: ")

y_gmm = pickle.load(open(os.getcwd() + '/pickles/gmm-' + puzzle,'rb'))
transf = pickle.load(open(os.getcwd() + '/pickles/pca-' + puzzle,'rb'))

plt.scatter(transf[:,0], transf[:,1],c=y_gmm, cmap='RdYlBu',s=150)
plt.suptitle("Puzzle " + puzzle,fontsize=18)
