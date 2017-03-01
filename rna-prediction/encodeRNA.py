# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 12:51:30 2017

@author: Rohan
"""

'''
encode RNA strucutre and encode movesets
'''
import pandas as pd
from readData import read_movesets
import os
def longest(a):
    return max(len(a), *map(longest, a)) if isinstance(a, list) and a else 0

def encode_structure(structure):
  encoded_structure = []
  for i in structure:
    if i == ".":
      encoded_structure.append([0])
    elif i == "(" or i == ")":
      encoded_structure.append([1])
  
  return encoded_structure
'''
def encode_movesets(moveset):
  encoded_ms = []
  for move in moveset:
    if move[0]['base'] == 'A':
      encoded_ms.append([1,move[0]['pos']])
    elif move[0]['base'] == 'U':
      encoded_ms.append([2,move[0]['pos']])
    elif move[0]['base'] == 'G':
      encoded_ms.append([3,move[0]['pos']])
    elif move[0]['base'] == 'C':
      encoded_ms.append([4,move[0]['pos']])
    
  return encoded_ms
'''
l = []
movesets = [[{'base':'G','pos':3}],[{'base':'A','pos':8},{'base':'U','pos':12}]]
for move in movesets:
  for i in move:
    pass
    
def encode_movesets(moveset):
    ms = []
    max_moves = longest(moveset)
    for k in moveset:
        #max_moves = len(max(k,key=len))
        #max_moves = longest(k)
        soln = []
        #if k[0][0]['type'] == 'reset':
         #   data_6892344.pop(data_6892344.index(k))
        for i in k:
            n_moves = len(i)
            i_moves = []
            for j in i:
                if 'type' in j:
                    i_moves.append([1,12345])
                elif j['base'] == 'A':
                    i_moves.append([1,j['pos']])
                elif j['base'] == 'U':
                    i_moves.append([2,j['pos']])
                elif j['base'] == 'G':
                    i_moves.append([3,j['pos']])
                elif j['base'] == 'C':
                    i_moves.append([4,j['pos']])
                elif j['type'] == 'paste' or j['type'] == 'reset':
                    continue
            #i_moves = [i_moves]
            i_moves = i_moves + (max_moves - n_moves)*[[0,0]]
            soln.append(i_moves)
        ms.append(soln)
        
    return ms


def encode_movesets_v3(moveset):
    ms = []
    #lens = [len(x) for j in x for x in moveset]
    #max_lens = max(lens)
    for k in moveset:
        player = []
        for i in k:
            for j in i:
                if 'type' in j:
                    player.append(1)
                    player.append(12345)
                elif j['base'] == 'A':
                    player.append(1)
                    player.append(j['pos'])
                elif j['base'] == 'U':
                    player.append(2)
                    player.append(j['pos'])
                elif j['base'] == 'G':
                    player.append(3)
                    player.append(j['pos'])
                elif j['base'] == 'C':
                    player.append(4)
                    player.append(j['pos'])
                elif j['type'] == 'paste' or j['type'] == 'reset':
                    continue
        ms.append(player)
    lens = [len(j) for j in ms]
    max_lens = max(lens)
    ms2 = []
    for l in ms:
        l.extend([0]*(max_lens-len(l)))
        
    
    return ms

'''
data_6892344 = read_movesets(os.getcwd() + '/movesets/move-set-11-14-2016.txt',6892344)
print len(data_6892344)
lens = [len(x) for x in j for j in data_6892344]
print lens
print sum(lens)

def encode_movesets_dataframe(moveset):
    
    pass

columns = ['pid','time','base','loc']
edf = pd.DataFrame(index=range(sum(lens)),columns=columns,dtype='float')
print edf
'''
'''
encoded_ms = []
for i in ms_6503049:
  print (i[0]['pos'])
  if i[0]['base'] == 'A':
    encoded_ms.append([1,i[0]['pos']])
  elif i[0]['base'] == 'U':
    encoded_ms.append([2,i[0]['pos']])
  elif i[0]['base'] == 'G':
    encoded_ms.append([3,i[0]['pos']])
  elif i[0]['base'] == 'C':
    encoded_ms.append([4,i[0]['pos']])

print encoded_ms

'''

