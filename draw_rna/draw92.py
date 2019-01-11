# IN PYTHON2

from subprocess import call
import os
import pandas as pd

path_to_movesets = os.getcwd()[:47] + '/rna-prediction/movesets/'

with open(path_to_movesets + 'teaching-puzzle-ids.txt') as f:
    content = f.readlines()

df = pd.read_csv(path_to_movesets + 'puzzle-structure-data.txt', sep=" ", header="infer", delimiter='\t')
df = df.set_index('pid')



teachingpuzzles = map(int, content)

for pid in teachingpuzzles:
    file = open(os.getcwd() + '/s4/p%i.txt' % pid, 'w')
    file.write(str(pid))
    struc = df.loc[pid, 'structure']
    file.write('\n' + 'A'*len(struc))
    file.write('\n' + struc)
    file.close()

    call(['python', 'draw_all.py', 's4/p%i.txt' % pid])

    print('########### Completed %i out of %i ############' % (teachingpuzzles.index(pid) + 1, len(teachingpuzzles)))