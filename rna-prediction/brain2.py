# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 13:42:03 2017

@author: rohankoodli
"""

import os
from readData import read_movesets_pid
from encodeRNA import encode_movesets
import numpy as np
import pandas as pd
import ast

filepath = os.getcwd() + '/movesets/move-set-11-14-2016.txt'
data, users = read_movesets_pid(filepath,6892348)
encoded = (encode_movesets(data))

moveset_dataFrame = pd.read_csv(filepath, sep=" ", header="infer", delimiter='\t')
puzzles_pid = (moveset_dataFrame.loc[moveset_dataFrame['pid'] == 6892348])
#print puzzles_pid

plist = list(puzzles_pid['move_set'])
#print plist
bf_list = []
for i in plist:
  s1 = (ast.literal_eval(i))
  s2 = s1['begin_from']
  bf_list.append(s2)


#print bf_list[0]

encoded_bf = []

for start in bf_list:
    enc = []
    for i in start:
        if i == 'A':
            enc.append(1)
        elif i == 'U':
            enc.append(2)
        elif i == 'G':
            enc.append(3)
        elif i == 'C':
            enc.append(4)
    encoded_bf.append(enc)

print encoded_bf[0]
print encoded[0]
