import os
from readData import read_movesets
from encodeRNA import encode_movesets
import numpy as np
from sklearn import mixture, decomposition
from matplotlib import pyplot as plt
import seaborn; seaborn.set()

data = read_movesets(os.getcwd() + '/movesets/move-set-11-14-2016.txt',7254761)
encoded = np.matrix(encode_movesets(data))

#print encoded_6892344[0]
print np.array(encoded).shape

#gmm = mixture.GMM()
#gmm.fit(encoded_6892344[0:5])

pca = decomposition.PCA(n_components=2)
pca.fit(encoded)
transf = pca.transform(encoded)
pc_variance = pca.explained_variance_ratio_

pc1 = str(pc_variance[0]*100)
pc1 = pc1[0:4]+'%'

pc2 = str(pc_variance[1]*100)
pc2 = pc2[0:4]+'%'

x,y = [],[]
for i in transf:
    x.append(i[0])
    y.append(i[1])

#plt.scatter(x,y)
#plt.show()

gmm = mixture.GaussianMixture(7)
gmm.fit(transf)
print gmm.bic(transf)
y_gmm = gmm.predict(transf)
plt.scatter(transf[:,0], transf[:,1],c=y_gmm,cmap='RdYlBu',s=150)
plt.suptitle("Puzzle 7254761",fontsize=18)
plt.xlabel('Component 1 (Explained Variance: %s)'%(pc1),fontsize=14)
plt.ylabel('Component 2 (Explained Variance: %s)'%(pc2),fontsize=14)

#plt.savefig(os.getcwd() + '/clustering-graphs/pid_7254761.pdf')
'''
reds,blues,yellows = 0,0,0

for i in y_gmm:
    if i == 0:
        reds += 1
    elif i == 1:
        yellows += 1
    elif i ==2:
        blues += 1
        
print 'Reds:',reds,'\nBlues:',blues,'\nYellows:',yellows

#print data_6892344[-2]

#print(data_6892344[0])
#print (data_6892344[4])
'''
'''
def longest(a):
    return max(len(a), *map(longest, a)) if isinstance(a, list) and a else 0
'''

'''
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
'''
'''
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
t2 = vec.fit_transform(data_6892344[0][0]).toarray()
print data_6892344[0][0]
'''

'''
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
'''
#movesets_6892344 = encode_movesets(data_6892344)
#print ms[1]
'''
test = [[[[3,10],[1,5],[3,18]],[[3,1],[0,0],[0,0]],[[3,3],[1,5],[0,0]]],[[[1,5],[2,7],[0,0]],[[1,7],[0,0],[0,0]],[[0,0],[0,0],[0,0]]]]
t2 = np.reshape(test,(1,-1))
print t2
from sklearn import mixture,cluster
gmm = mixture.GMM()
kmm = cluster.KMeans(n_clusters=8)
kmm.fit(t2)
'''
'''
test_ms = np.array([[[[3,1],[0,0],[0,0],[0,0]],[[3,3],[1,5],[0,0],[0,0]]],[[[1,5],[2,7],[0,0],[0,0]],[[1,7],[0,0],[0,0],[0,0]]]])
test = [[[[3,10],[1,5],[3,18]],[[3,1],[0,0],[0,0]],[[3,3],[1,5],[0,0]]],[[[1,5],[2,7],[0,0]],[[1,7],[0,0],[0,0]],[[0,0],[0,0],[0,0]]]]
test = np.array(test)
from sklearn import cluster, mixture, decomposition, datasets
#pca = decomposition.PCA().fit(test)
gmm = mixture.GMM()
test = np.array(test)
print test.shape
move1 = [[2,5],[3,7],[3,8]]
gmm.fit(move1)
iris = datasets.load_iris()
X = iris.data

kmm = cluster.KMeans()
kmm.fit_transform(test)
'''
'''
moveset_df = pd.DataFrame(movesets_6892344)
moveset_df.to_csv(os.getcwd()+'/movesets/encoded-movesets.csv',sep=',')
'''

#from sklearn import mixture,cluster,decomposition
#from sklearn.feature_selection import chi2,SelectKBest
#X_new = SelectKBest(chi2, k=2).fit_transform(test_ms)



'''
pca = decomposition.PCA(n_components=2)
pca.fit(test_ms,4)
pca.transform(test_ms)
#X = pca.transform(ms)
#kmm = cluster.k_means(,3)
gmm = mixture.GMM()
#kmm.fit([[[[3,1],[0,0],[0,0],[0,0]],[[3,3],[1,5],[0,0],[0,0]]],[[[1,5],[2,7],[0,0],[0,0]],[[1,7],[0,0],[0,0],[0,0]]]])

#kmm = cluster.k_means(t2,2)
'''


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

