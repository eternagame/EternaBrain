# -*- coding: utf-8 -*-
"""
Created on Thu Jan 05 18:30:26 2017

@author: Rohan
"""
import pandas as pd

def getData_pid(pid,pidList,movesets,structure): # returns moveset and puzzzle structure together
  i1 = pidList.index(pid)
  #return movesets[num]
  pid_structure = structure['pid']
  pid_puzzleList = list(pid_structure)
  i2 = pid_puzzleList.index(pid)
  
  return movesets[i1], structure['structure'][i2]

  
  
  