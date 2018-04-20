import pandas as pd
from pandas import ExcelWriter
import os, ast

uidList = [267, 1623, 2577, 2804, 4167, 4375, 8627, 11775, 19442, 24263,
           26574, 28158, 28979, 29762, 32487, 32627, 34334, 34596, 35880,
           36921, 39309, 40516, 41429, 42101, 42404, 42833, 42966, 43776,
           44066, 44191, 44631, 46281, 48166, 48170, 50065, 52361, 53165,
           55082, 55300, 56579, 57654, 57675, 57743, 57874, 58224, 60391,
           61031, 61244, 61455, 64544, 65056, 72033, 77611, 78801, 82881,
           87216, 91837, 132701, 133043, 136909, 143547, 148829, 152013,
           179978, 202762, 207838, 209752, 216078, 223393, 225023, 231977,
           233206]

def stats(pidList, uidList):
    moveset_dataFrame = pd.read_csv(os.getcwd() + '/movesets/moveset6-22a.txt', sep=" ", header="infer", delimiter='\t')
    num_sols = []
    num_moves = []
    for uid in uidList:
        sum_moves = 0
        #for pid in pidList:
        puzzles1 = moveset_dataFrame.loc[moveset_dataFrame['uid'] == uid]
        # puzzles2 = puzzles1.loc[puzzles1['pid'] == pid]
        ms = list(puzzles1['move_set'])
        for i in ms:
            #print i
            #print type(i)
            try:
                i = ast.literal_eval(i)
            except ValueError:
                continue
            #print i['num_moves']
            sum_moves += int(i['num_moves'])

        if len(ms) > 0:
            num_sols.append(len(ms))
            num_moves.append(sum_moves)
        else:
            num_sols.append(0)
            num_moves.append(0)

        print('Completed %i out of %i' %(uidList.index(uid) + 1, len(uidList)))

    print len(uidList),len(num_sols),len(num_moves)
    d = {'User_IDs': uidList, 'Num_solutions': num_sols, 'Num_moves': num_moves}
    df = pd.DataFrame(data=d)

    return df

def convert():
    writer = ExcelWriter(os.getcwd() + '/movesets/supplementaltable1.xlsx')

    old_df = pd.read_excel(os.getcwd() + '/movesets/old_supplementaltable1.xlsx')
    old_df['IDs'] = uidList
    for i, r in old_df[::-1].iterrows():
        print i, r['User_IDs']
        if r['Num_moves'] == 0:
            old_df = old_df.drop([i])

    old_df.to_excel(writer)

def all_uids():
    writer = ExcelWriter(os.getcwd() + '/movesets/supplementaltable2.xlsx')
    moveset_dataFrame = pd.read_csv(os.getcwd() + '/movesets/moveset6-22a.txt', sep=" ", header="infer", delimiter='\t')

    puzzles1 = list(moveset_dataFrame['uid'].values)
    puzzles2 = list(moveset_dataFrame['uid'].unique())
    print len(puzzles1)
    print len(puzzles2)
    print len(set(puzzles1))
    f = open(os.getcwd() + '/movesets/supplementaltable2.txt', 'wb')
    for i in list(set(puzzles1)):
        f.write("%i\n" % i)
    f.close()


if __name__ == '__main__':
    all_uids()
