import os
file = os.getcwd() + '\movesets\epicfalcon.txt'

import pandas as pd
epicfalcon = pd.read_csv(file, sep=" ", header='infer', delimiter='\t')
# epicfalcon is a dataframe containging epicfalcon.txt data

ms1 = epicfalcon[['move_set']].ix[[1]]
# ms1 is moveset for pid 6502951
ms2 = ms1.to_dict()
# ms2 is dictionary of ms1
ms3 = (ms2['move_set'])
# ms3 is moveset of ms2
ms4 = ms3[1]
# ms4 is ms3 w/o labels

import ast

ms5 = ast.literal_eval(ms4)
#print (ms5['moves'])
# ms5 is dict of ms4
# can now be indexed like a normal dictionary

movesets = [] # a list of dictionaries containing the movesets
for i in range(101):
    step1 = epicfalcon[['move_set']].ix[[i]]
    step2 = step1.to_dict()
    step3 = step2['move_set']
    step4 = step3[i]
    step5 = ast.literal_eval(step4)
    movesets.append(step5['moves'])
    
