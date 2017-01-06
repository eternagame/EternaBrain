import os
from movesetreader import read_movesets, puzzle_attributes
from structure import read_structure
import pandas as pd

# read moveset file
# 102 total movesets
epicfalcon = os.getcwd() + '\movesets\epicfalcon.txt'

movesets = read_movesets(epicfalcon)
pid = puzzle_attributes(epicfalcon,'pid') #puzzle ID
sol_id = puzzle_attributes(epicfalcon,'sol_id') # solution ID
uid = puzzle_attributes(epicfalcon,'uid') # user ID

# read puzzle data file
puzzle_structure_data = os.getcwd() + '\movesets\puzzle-structure-data.txt'

structure = read_structure(puzzle_structure_data)

# The puzzle ID, solution ID, and moveset for the final puzzle in epicfalcon.txt
print pid[100]
print sol_id[100]
print uid[100] # uid will be the same as they are only 1 player's solutions
print movesets[100]

print type(movesets)
ms = pd.Series(movesets)


print type(structure['pid'])
pid2 = structure['pid']

print pid2[pid2==6503049]
print structure['structure'][19195]

'''
pid2 = pd.Series(pid)
print type(pid2)

print pid2[pid2==6503049]
'''
#print(movesets[0][0][0]['pos']) # first puzzle, first move, position of move
#print(movesets[15][24][0]['base']) # 16th puzzle, 25th move, base(AUGC) of move
#print(structure['structure'][20672]) # 20,673 solutions

