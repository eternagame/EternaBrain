# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 12:51:30 2017

@author: Rohan
"""

'''
encode RNA strucutre and encode movesets
'''

def encode_structure(structure):
  encoded_structure = []
  for i in structure:
    if i == ".":
      encoded_structure.append([0])
    elif i == "(" or i == ")":
      encoded_structure.append([1])
  
  return encoded_structure

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