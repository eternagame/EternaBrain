# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 13:42:03 2017

@author: rohankoodli
"""

import os
from readData import read_movesets_pid
from encodeRNA import encode_movesets, encode_movesets_style
import numpy as np
import pandas as pd
import ast

filepath = os.getcwd() + '/movesets/move-set-11-14-2016.txt'
data, users = read_movesets_pid(filepath,6892348)
encoded = (encode_movesets_style(data))

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

encoded_bf = [] # this is input for getting structure from Vienna webserver

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

#print encoded_bf[0]
#print len(encoded[0])

for i in ((encoded)):
    for j in range(len(i)):
        pass

X,y = [],[]


for i,j in (zip(encoded_bf,encoded)):
    print i
    for m in j:
        #print m[1]
        X.append(i)
        y.append(m)
        loc = m[1] - 1
        #i = i[loc].replace(m[0])
        i[loc] = m[0]

#print X[0]
#print X[110]
#print X[114]
#print len(X)
#print len(y)

X2,y2 = [],[]

while i in range(len(encoded_bf)):
    pass



