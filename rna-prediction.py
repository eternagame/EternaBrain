import os
file = os.getcwd() + '\movesets\epicfalcon.txt'

import pandas as pd
epicfalcon = pd.read_csv(file, sep=" ", header='infer', delimiter='\t')
print epicfalcon

