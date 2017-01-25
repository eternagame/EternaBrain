import os
from movesetreader import read_movesets, puzzle_attributes
from structure import read_structure
from getData import getData_pid

# read moveset file
# 102 total movesets
epicfalcon = os.getcwd() + '\movesets\epicfalcon.txt'
epicfalcon = os.getcwd().replace("\RNA-Prediction","") + '\RNA-Prediction\movesets\epicfalcon.txt'


movesets = read_movesets(epicfalcon)
pid = puzzle_attributes(epicfalcon,'pid') #puzzle ID
sol_id = puzzle_attributes(epicfalcon,'sol_id') # solution ID
uid = puzzle_attributes(epicfalcon,'uid') # user ID

# read puzzle data file
puzzle_structure_data = os.getcwd().replace('\RNA-Prediction','') + '\RNA-Prediction\movesets\puzzle-structure-data.txt'

structure = read_structure(puzzle_structure_data)


ms_6503049,stctr_6503049 = getData_pid(6503049,pid,movesets,structure)
ms_6502960,stctr_6502960 = getData_pid(6502960,pid,movesets,structure)




'''
# The puzzle ID, solution ID, and moveset for the final puzzle in epicfalcon.txt
print pid[100]
print sol_id[100]
print uid[100] # uid will be the same as they are only 1 player's solutions
print movesets[100]
'''
'''
print pid.index(6503049)


print type(movesets)
ms = pd.Series(movesets)

print type(pid)
pidd = pd.Series(pid)
agar = pidd[pidd==6503049]


print type(structure['pid'])
pid2 = structure['pid']

print pid2[pid2==6503049]
print structure['structure'][19195]
print '----------------------------------------'
print type(pid2)
p3 = list(pid2)
print type(p3)
print p3.index(6503049)
'''
'''
pid2 = pd.Series(pid)
print type(pid2)

print pid2[pid2==6503049]
'''
#print(movesets[0][0][0]['pos']) # first puzzle, first move, position of move
#print(movesets[15][24][0]['base']) # 16th puzzle, 25th move, base(AUGC) of move
#print(structure['structure'][20672]) # 20,673 solutions

