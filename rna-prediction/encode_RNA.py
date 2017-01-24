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
  pass


