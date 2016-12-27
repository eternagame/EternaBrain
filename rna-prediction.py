import os
import pandas as pd
import ast
from movesetreader import read_movesets

f = os.getcwd() + '\movesets\epicfalcon.txt'
epicfalcon = pd.read_csv(f, sep=" ", header='infer', delimiter='\t')
# epicfalcon is a dataframe containging epicfalcon.txt data

movesets = read_movesets(epicfalcon)

print(movesets[0][0][0]['pos']) # first puzzle, first move, position of move
print(movesets[15][24][0]['base']) # 16th puzzle, 25th move, base(AUGC) of move


