'''
Loads the saved TensorFlow models and runs a simulation of EternaBrain solving a real Eterna puzzle
Input a target structure in dot-bracket notation and any initial params (energy, locked bases)
'''

import tensorflow as tf
import os
import pickle
import numpy as np
import RNA
import copy
from numpy.random import choice
from difflib import SequenceMatcher
from readData import format_pairmap
from sap1 import sbc
from sap2 import dsp
import pandas as pd
from predict_pm import predict


LOCATION_FEATURES = 6
BASE_FEATURES = 7
NAME = 'CNN20'

if __name__ == '__main__':
    p = pd.read_csv(os.getcwd()+'/movesets/eterna100.txt', sep=' ', header='infer', delimiter='\t')
    plist = list(p['Secondary Structure'])

    num_completed, num_solved = 0, 0
    with open('predict100_results.txt', 'r+') as f:
        contents = f.read()
        num_completed = int(contents[-1])
        num_solved = int(contents[-3])

    while num_completed <= 100:
        solved = predict(plist[num_completed], False)
        num_completed += 1
        if solved:
            num_solved += 1
        with open('predict100_results.txt', 'w+') as f:
            f.write('\nSolved %i/%i' % (num_solved, num_completed))
        print('Solved %i/%i' % (num_solved, num_completed))
