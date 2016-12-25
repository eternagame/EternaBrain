import os
file = os.getcwd() + '\movesets\epicfalcon.txt'

import pandas as pd
epicfalcon = pd.read_csv(file, sep=" ", header='infer', delimiter='\t')
# epicfalcon is a dataframe containging epicfalcon.txt data

ms1 = epicfalcon[['move_set']].ix[[0]]

ms2 = ms1.to_dict()

ms3 = (ms2['move_set'])

ms4 = ms3[0]

#print(ms4)

import ast

ms5 = ast.literal_eval(ms4)

print (ms5['moves'][0])