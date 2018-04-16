# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 12:57:43 2016

@author: Rohan
"""
import ast
import pandas as pd
import os
import numpy as np
from eterna_score import get_pairmap_from_secstruct

def experience(threshold):
    '''
    Returns player IDs of players who have solved over a certain threshold of puzzles

    :param threshold: The minimum number of puzzles to be classified as an "expert"
    :return: A list of "expert" players
    '''
    full_problems = pd.read_csv(os.getcwd()+'/movesets/full-problems-nov2016.txt', sep=" ", header="infer", delimiter='\t')
    user_df = full_problems[['uid']]
    users = np.array(user_df,dtype=int) # list of users
    experienced_players = []
    unique, counts = np.unique(users, return_counts=True) # unique and counts are same length

    for i in range(len(unique)):
        if counts[i] >= threshold:
            experienced_players.append(unique[i])

    return experienced_players


def read_movesets_uid(uid): # get data from user ID
    '''
    Returns move sets from a specific player

    :param uid: The user ID of the Eterna player
    :return: The move sets of that player
    :return: The list of puzzle IDs of puzzles that he/she has solved
    '''
    moveset_dataFrame = pd.read_csv(os.getcwd()+'/movesets/moveset6-22a.txt', sep=" ", header="infer", delimiter='\t')
    puzzles_uid = moveset_dataFrame.loc[moveset_dataFrame['uid'] == uid]
    plist = list(puzzles_uid['move_set'])
    pidList = list(puzzles_uid['pid'])
      # plist_dict = []
      # for i in plist:
      #   s1 = (ast.literal_eval(i))
      #   s2 = s1['moves']
      #   plist_dict.append(s2)

    return plist, pidList


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


def read_movesets_uid_pid(uid,pid,df='list'): # get data from user ID
    """
    Returns move sets for a specific puzzle only 1 user has solved

    :param uid: The user's ID
    :param pid: The puzzle ID
    :param df: Whether you want the returned move sets to be in a pandas dataframe or a list
    :return: The move sets either in a list or a pandas dataframe
    """
    moveset_dataFrame = pd.read_csv(os.getcwd()+'/movesets/moveset6-22a.txt', sep=" ", header="infer", delimiter='\t')
    puzzles1 = moveset_dataFrame.loc[moveset_dataFrame['uid'] == uid]
    puzzles2 = puzzles1.loc[puzzles1['pid'] == pid]
    if df == "list":
        return list(puzzles2['move_set'])
    elif df == "df":
        return puzzles2

    # plist = list(puzzles2)
    # plist_dict = []
    # for i in plist:
    #   s1 = (ast.literal_eval(i))
    #   s2 = s1['moves']
    #   plist_dict.append(s2)
    #
    # return plist_dict

'''
DEPRECATED
Previous version of the experience method
'''


def experience_labs(puzzle,threshold):
    experience_file = pd.read_csv(os.getcwd()+'/movesets/prior-experience-labs.txt', sep=' ', header='infer', delimiter=',')
    moveset_file = pd.read_csv(os.getcwd()+'/movesets/moveset6-22a.txt', sep=' ', header='infer', delimiter='\t')
    # puzzles_pid = experience_file.loc[experience_file['pid'] == puzzles]
    # spec = puzzles_pid.loc[puzzles_pid['prior_puzzle'] > threshold]
    # user_list = list(spec['uid'])
    # user_list = list(set(user_list))
    #
    # return user_list
    user_list = []
    lens = []
    #for puzzle in puzzles:
    puzzles_pid = experience_file.loc[experience_file['pid'] == puzzle]
    spec = puzzles_pid.loc[puzzles_pid['prior_puzzle'] > threshold]
    puzzle_nums = len(list(spec['pid']))
    puzzle_players = list(spec['uid'])
    puzzle_players = list(set(puzzle_players))
    user_list.extend(puzzle_players)
    lens.append(puzzle_nums)
    #print user_list
    user_list = list(set(user_list))
    final_dict = []
    bf_list = []
    #for puzzle in (puzzles):
    for user in user_list:
        n = read_movesets_uid_pid(user,puzzle)
        for i in n:
            s1 = ast.literal_eval(i)
            s2 = s1['moves']
            s3 = s1['begin_from']
            final_dict.append(s2)
            bf_list.append(s3)

    encoded_bf = []
    for start in bf_list:
       enc = []
       for i in start:
           if i == 'A':
               enc.append(1)
           elif i == 'U':
               enc.append(2)
           elif i == 'G':
               enc.append(3)
           elif i == 'C':
               enc.append(4)
       encoded_bf.append(enc)

    return final_dict,user_list,encoded_bf,lens


def read_locks(pid):
    '''
    Returns a list of locked bases (0s and 1s) for a specific puzzle

    :param pid: The puzzle ID
    :return: The locked bases in a list of 0s and 1s
    '''
    puzzle_structure = pd.read_csv(os.getcwd()+'/movesets/puzzle-structure-data.txt', sep=" ", header='infer', delimiter='\t')
    puzzles_pid = puzzle_structure.loc[puzzle_structure['pid'] == pid]

    try:
        str_locks = ''.join(list(puzzles_pid['locks']))
        enc_locks = []
        for k in str_locks:
            if k == 'o':
                enc_locks.append(1) #unlocked
            elif k == 'x':
                enc_locks.append(2) #locked

        return enc_locks

    except TypeError:
        return "None"


def read_movesets_pid(moveset_file,pid): # get data from puzzle ID
    '''
    Reads and returns the move sets for a certain puzzle

    :param moveset_file: The filepath to the move set data
    :param pid: The puzzle ID of the puzzle
    :return: The list of the movesets for that specific puzzle
    :return: The list of users who have solved that puzzle
    '''
    moveset_dataFrame = pd.read_csv(moveset_file, sep=" ", header="infer", delimiter='\t')
    puzzles_pid = moveset_dataFrame.loc[moveset_dataFrame['pid'] == pid]
    plist = list(puzzles_pid['move_set'])
    ulist = list(puzzles_pid['uid'])
    plist_dict = []
    for i in plist:
        s1 = (ast.literal_eval(i))
        s2 = s1['moves']
        plist_dict.append(s2)

    return plist_dict, ulist


def read_structure(pid):
    '''
    Gets the structure of the RNA puzzle

    :param: pid: The puzzle ID
    :return: The structure encoded from dot-bracket notation into 1s, 2s and 3s
    '''
    puzzle_structure = pd.read_csv(os.getcwd()+'/movesets/puzzle-structure-data.txt', sep=" ", header='infer', delimiter='\t')
    puzzles_pid = puzzle_structure.loc[puzzle_structure['pid'] == pid]

    str_struc = ''.join(list(puzzles_pid['structure']))
    enc_struc = []
    for k in str_struc:
        if k == '.':
            enc_struc.append(1)
        elif k == '(':
            enc_struc.append(2)
        elif k == ')':
            enc_struc.append(3)

    return enc_struc

def structure_avg(pidList):
    puzzle_structure = pd.read_csv(os.getcwd()+'/movesets/puzzle-structure-data.txt', sep=" ", header='infer', delimiter='\t')
    avgs = []
    for pid in pidList:
        puzzles_pid = puzzle_structure.loc[puzzle_structure['pid'] == pid]
        str_struc = ''.join(list(puzzles_pid['structure']))
        avgs.append(1.0/len(str_struc))

    return sum(avgs)/float(len(pidList))

def format_pairmap(struc):
    pm = get_pairmap_from_secstruct(struc)
    new = []
    for i in pm:
        if i == -1:
            new.append(i)
        else:
            i += 1
            new.append(i)

    return pm


def read_structure_raw(pid):
    puzzle_structure = pd.read_csv(os.getcwd()+'/movesets/puzzle-structure-data.txt', sep=" ", header='infer', delimiter='\t')
    puzzles_pid = puzzle_structure.loc[puzzle_structure['pid'] == pid]

    struc = ''.join(list(puzzles_pid['structure']))
    return struc


def read_movesets_all(moveset_file): # get data from puzzle ID
  moveset_dataFrame = pd.read_csv(moveset_file, sep=" ", header="infer", delimiter='\t')
  puzzles_pid = moveset_dataFrame.loc[moveset_dataFrame['pid']]
  plist = list(puzzles_pid['move_set'])
  #ulist = list(puzzles_pid['uid'])
  plist_dict = []
  for i in plist:
    s1 = (ast.literal_eval(i))
    s2 = s1['moves']
    plist_dict.append(s2)

  return plist_dict


def puzzle_attributes(moveset_file, attribute):
  moveset_dataFrame = pd.read_csv(moveset_file, sep=" ", header="infer", delimiter='\t')
  attribute_list = []
  for i in range(101):
    step1 = moveset_dataFrame[[attribute]].ix[[i]]
    step2 = step1[attribute]
    step3 = step2[i]
    step4 = str(step3)
    step5 = int(step4)
    attribute_list.append(step5)

  return attribute_list


def read_structure_all(puzzle_data):
  puzzle_structure = pd.read_csv(puzzle_data, sep=" ", header='infer', delimiter='\t')

  return puzzle_structure

#####################################################################

'''
def read_movesets_v0(moveset_file):
  moveset_dataFrame = pd.read_csv(moveset_file, sep=" ", header="infer", delimiter='\t')
  movesets = [] # a list of dictionaries containing the movesets
  for i in range(len(moveset_dataFrame)): # 102 total moveset solutions in epicfalcon.txt
      step1 = moveset_dataFrame[['move_set']].ix[[i]] # str of pid, sol_id, uid, and moveset
      step2 = step1.to_dict() # dictionary of data
      step3 = step2['move_set'] # selecting only moveset data
      step4 = step3[i] # getting rid of labels
      step5 = ast.literal_eval(step4) # converting movesets to dictionary
      movesets.append(step5['moves']) # adding each moveset to list

  return movesets
'''

'''
complete = os.getcwd() + '/movesets/move-set-11-14-2016.txt'
complete = pd.read_csv(complete, sep=" ", header='infer', delimiter='\t')
puzzles_6892344 = complete.loc[complete['pid'] == 6892344]
#print puzzles_6892344['move_set']
plist = list(puzzles_6892344['move_set'])
print type(plist[0])
plist_dict = []
for i in plist:
  s1 = (ast.literal_eval(i))
  s2 = s1['moves']
  plist_dict.append(s2)

print (plist_dict[25])
'''

'''taking pid and making a list of all the puzzle ID's in a list with indexes corresponding to movesets
epicfalcon = os.getcwd() + '\movesets\epicfalcon.txt'
epicfalcon_dataframe = pd.read_csv(epicfalcon, sep=' ', header='infer', delimiter='\t')
a = epicfalcon_dataframe[['pid']].ix[[1]]
b = a['pid']
print b
c = b[1]
print c
d = str(c)
e = int(d)
print (type(e))
'''

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

'''
f = os.getcwd() + '\movesets\puzzle-structure-data.txt'
puzzle_structure = pd.read_csv(f, sep=" ", header='infer', delimiter='\t')

print puzzle_structure['pid'][10]
'''
'''
ms1 = puzzle_structure[['structure']].ix[[0]]
# ms1 is moveset for pid 6502951
ms2 = ms1.to_dict()
# ms2 is dictionary of ms1
ms3 = (ms2['structure'])
# ms3 is moveset of ms2
ms4 = ms3[0]
# ms4 is ms3 w/o labels


ms5 = ast.literal_eval(ms4)
#print (ms5['moves'])
'''
