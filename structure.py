# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 11:01:32 2016

@author: Rohan
"""
import ast
import os
import pandas as pd
def read_structure(puzzle_data):
  pass


f = os.getcwd() + '\movesets\puzzle-structure-data.txt'
puzzle_structure = pd.read_csv(f, sep=" ", header='infer', delimiter='\t')

print puzzle_structure['pid'][10]

'''
ms1 = puzzle_structure[['structure']].ix[[0]]
# ms1 is moveset for pid 6502951
ms2 = ms1.to_dict()
# ms2 is dictionary of ms1
ms3 = (ms2['structure'])
# ms3 is moveset of ms2
ms4 = ms3[0]
# ms4 is ms3 w/o labels


ms5 = ast.literal_eval(ms4)
#print (ms5['moves'])
'''
