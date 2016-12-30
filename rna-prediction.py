import os
from movesetreader import read_movesets, get_puzzleData
from structure import read_structure

# read moveset file
# 102 total movesets
epicfalcon = os.getcwd() + '\movesets\epicfalcon.txt'

movesets = read_movesets(epicfalcon)
pid = get_puzzleData(epicfalcon,'pid')
print pid[100]

#print(movesets[0][0][0]['pos']) # first puzzle, first move, position of move
#print(movesets[15][24][0]['base']) # 16th puzzle, 25th move, base(AUGC) of move

# read puzzle structure data file
puzzle_structure_data = os.getcwd() + '\movesets\puzzle-structure-data.txt'

structure = read_structure(puzzle_structure_data)

#print(structure['structure'][20672]) # 20,673 solutions

