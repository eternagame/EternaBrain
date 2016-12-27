import os
import pandas as pd
import ast

file = os.getcwd() + '\movesets\epicfalcon.txt'
epicfalcon = pd.read_csv(file, sep=" ", header='infer', delimiter='\t')
# epicfalcon is a dataframe containging epicfalcon.txt data

''' example of converting moveset data to dictionary
ms1 = epicfalcon[['move_set']].ix[[1]]
# ms1 is moveset for pid 6502951
ms2 = ms1.to_dict()
# ms2 is dictionary of ms1
ms3 = (ms2['move_set'])
# ms3 is moveset of ms2
ms4 = ms3[1]
# ms4 is ms3 w/o labels


ms5 = ast.literal_eval(ms4)
#print (ms5['moves'])
# ms5 is dict of ms4
# can now be indexed like a normal dictionary
'''

movesets = [] # a list of dictionaries containing the movesets
for i in range(101): # 102 total moveset solutions in epicfalcon.txt
    step1 = epicfalcon[['move_set']].ix[[i]] # str of pid, sol_id, uid, and moveset
    step2 = step1.to_dict() # dictionary of data
    step3 = step2['move_set'] # selecting only moveset data
    step4 = step3[i] # getting rid of labels
    step5 = ast.literal_eval(step4) # converting movesets to dictionary
    movesets.append(step5['moves']) # adding each moveset to list
    
