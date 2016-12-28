import os
from movesetreader import read_movesets

epicfalcon = os.getcwd() + '\movesets\epicfalcon.txt'

movesets = read_movesets(epicfalcon)

print(movesets[0][0][0]['pos']) # first puzzle, first move, position of move
print(movesets[15][24][0]['base']) # 16th puzzle, 25th move, base(AUGC) of move


