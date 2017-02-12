import os
from readData import read_movesets
from encodeRNA import encode_movesets

def longest(a):
    return max(len(a), *map(longest, a)) if isinstance(a, list) and a else 0

test = read_movesets(os.getcwd() + '/movesets/move-set-11-14-2016.txt',6892344)[15]
max_len = len(max(test,key=len))
movesets = []
for i in test:
    num_moves = len(i)
    ind_moves = []
    for j in i:
        if j['base'] == 'A':
            ind_moves.append([1,j['pos']])
        elif j['base'] == 'U':
            ind_moves.append([2,j['pos']])
        elif j['base'] == 'G':
            ind_moves.append([3,j['pos']])
        elif j['base'] == 'C':
            ind_moves.append([4,j['pos']])
    ind_moves = [ind_moves]
    ind_moves = ind_moves + (max_len - num_moves)*[[0,0]]
    movesets.append(ind_moves)

19*[[0,0]] + [[2,5],[1,20]] #this works
#print movesets[len(movesets)-1]

data_6892344 = read_movesets(os.getcwd() + '/movesets/move-set-11-14-2016.txt',6892344)
#print data_6892344
ms = []
for k in data_6892344:
    #max_moves = len(max(k,key=len))
    max_moves = longest(k)
    soln = []
    #if k[0][0]['type'] == 'reset':
     #   data_6892344.pop(data_6892344.index(k))
    for i in k:
        n_moves = len(i)
        i_moves = []
        for j in i:
            if 'type' in j:
                i_moves.append([1,12345])
            elif j['base'] == 'A':
                i_moves.append([1,j['pos']])
            elif j['base'] == 'U':
                i_moves.append([2,j['pos']])
            elif j['base'] == 'G':
                i_moves.append([3,j['pos']])
            elif j['base'] == 'C':
                i_moves.append([4,j['pos']])
            elif j['type'] == 'paste' or j['type'] == 'reset':
                continue
        #i_moves = [i_moves]
        i_moves = i_moves + (max_moves - n_moves)*[[0,0]]
        soln.append(i_moves)
    ms.append(soln)
    
#print ms[1]

from sklearn import mixture,cluster
kmm = cluster.k_means([[[[3,1],[0,0],[0,0],[0,0]],[[3,3],[1,5],[0,0],[0,0]]],[[[1,5],[2,7],[0,0],[0,0]],[[1,7],[0,0],[0,0],[0,0]]]],3)
gmm = mixture.GMM()
kmm.fit([[[[3,1],[0,0],[0,0],[0,0]],[[3,3],[1,5],[0,0],[0,0]]],[[[1,5],[2,7],[0,0],[0,0]],[[1,7],[0,0],[0,0],[0,0]]]])
'''
# read moveset file
# 102 total movesets

epicfalcon = os.getcwd() + '/movesets/epicfalcon.txt'

movesets = read_movesets(epicfalcon)
pid = puzzle_attributes(epicfalcon,'pid') #puzzle ID
sol_id = puzzle_attributes(epicfalcon,'sol_id') # solution ID
uid = puzzle_attributes(epicfalcon,'uid') # user ID

# read puzzle data file
puzzle_structure_data = os.getcwd() + '/movesets/puzzle-structure-data.txt'

structure = read_structure(puzzle_structure_data)

ms_6503049,stctr_6503049 = getData_pid(6503049,pid,movesets,structure)
ms_6502960,stctr_6502960 = getData_pid(6502960,pid,movesets,structure)

encoded_6503049 = encode_movesets(ms_6503049)
'''
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

