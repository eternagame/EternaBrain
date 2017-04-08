# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 12:18:56 2017

@author: rohankoodli
"""


import os
import pandas as pd

filepath = os.getcwd() + '/movesets/move-set-11-14-2016.txt'
ms_df = pd.read_csv(filepath, sep=" ", header="infer", delimiter='\t')

#print ms_df
pidlist = list(ms_df['pid'])
individual_pids = list(set(pidlist))
print len(individual_pids)

over = 0
midhigh = 0
midlow = 0
less = 0
low = 0
verylow = 0

for i in individual_pids:
    counts = pidlist.count(i)
    if counts >= 5000: # 4
        over += 1
    elif counts < 5000 and counts >= 3000: # 19
        midhigh += 1
    elif counts < 3000 and counts >= 1000: # 35
        midlow += 1
    elif counts < 1000 and counts >= 500: # 26
        less += 1
    elif counts < 500 and counts >= 100: # 165
        low += 1
    elif counts < 100: # 8952
        verylow += 1

print over
print midhigh
print midlow
print less
print low
print verylow

