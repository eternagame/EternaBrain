# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 12:57:43 2016

@author: Rohan
"""
import ast
import pandas as pd
import os

def read_movesets(moveset_file):
  moveset_dataFrame = pd.read_csv(moveset_file, sep=" ", header="infer", delimiter='\t')
  movesets = [] # a list of dictionaries containing the movesets
  for i in range(101): # 102 total moveset solutions in epicfalcon.txt
      step1 = moveset_dataFrame[['move_set']].ix[[i]] # str of pid, sol_id, uid, and moveset
      step2 = step1.to_dict() # dictionary of data
      step3 = step2['move_set'] # selecting only moveset data
      step4 = step3[i] # getting rid of labels
      step5 = ast.literal_eval(step4) # converting movesets to dictionary
      movesets.append(step5['moves']) # adding each moveset to list
      
  return movesets
  
def puzzle_attributes(moveset_file, attribute):
  moveset_dataFrame = pd.read_csv(moveset_file, sep=" ", header="infer", delimiter='\t')
  attribute_list = []
  for i in range(101):
    step1 = moveset_dataFrame[[attribute]].ix[[i]]
    step2 = step1[attribute]
    step3 = step2[i]
    step4 = str(step3)
    step5 = int(step4)
    attribute_list.append(step5)
    
  return attribute_list

'''taking pid and making a list of all the puzzle ID's in a list with indexes corresponding to movesets
epicfalcon = os.getcwd() + '\movesets\epicfalcon.txt'
epicfalcon_dataframe = pd.read_csv(epicfalcon, sep=' ', header='infer', delimiter='\t')
a = epicfalcon_dataframe[['pid']].ix[[1]]
b = a['pid']
print b
c = b[1]
print c
d = str(c)
e = int(d)
print (type(e))
'''

''' example of converting moveset data to dictionary
ms1 = epicfalcon[['move_set']].ix[[1]]
# ms1 is moveset for pid 6502951
ms2 = ms1.to_dict()
# ms2 is dictionary of ms1
ms3 = (ms2['move_set'])
# ms3 is moveset of ms2
ms4 = ms3[1]
# ms4 is ms3 w/o labels


ms5 = ast.literal_eval(ms4)
#print (ms5['moves'])
# ms5 is dict of ms4
# can now be indexed like a normal dictionary
'''
