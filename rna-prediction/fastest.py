'''
finds fastest solutions
'''

import numpy as np
import os
from readData import experience_labs, experience, read_movesets_uid_pid
from encodeRNA import base_sequence_at_current_time_pr, structure_and_energy_at_current_time
from encodeRNA import encode_bases,encode_location,encode_movesets_style_pr
from encodeRNA import structure_and_energy_at_current_time_with_location
import ast
import time
import pickle
import pandas as pd

with open(os.getcwd()+'/movesets/teaching-puzzle-ids.txt') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]
content = [int(x) for x in content]
progression = [6502966,6502968,6502973,6502976,6502984,6502985,6502993, \
                6502994,6502995,6502996,6502997,6502998,6502999,6503000] # 6502957
content.extend(progression)
content.remove(6502993)
max_moves = 30

len_longest = 500

new_ms = os.getcwd() + '/movesets/moveset6-22a.txt'
moveset_dataFrame = pd.read_csv(new_ms, sep=' ', header='infer', delimiter='\t')

def speed(pid):
    """
    Encodes the puzzle solutions that were completed in the fewest
    number of moves

    :param pid: Puzzle ID
    :return: CNN training data of fastest solutions for that puzzle
    """
    final_dict = []
    bf_list = []

    #for pid in pidList:
    print pid
    puzzles_pid = moveset_dataFrame.loc[moveset_dataFrame['pid'] == pid]
    plist = list(puzzles_pid['move_set'])
    ulist = list(puzzles_pid['uid'])
    plist_dict = []
    for i in (plist):
        s1 = (ast.literal_eval(i))
        s2 = int(s1['num_moves'])
        if s2 <= max_moves: # solved in 50 moves or less
            print 'fast'
            s3 = s1['moves']
            s4 = s1['begin_from']
            final_dict.append(s3)
            bf_list.append(s4)
        else:
            continue

    print "complete data read"
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
    print "encoded begin_from"
    print len(final_dict)
    encoded = encode_movesets_style_pr(final_dict)
    encoded_base = (encode_bases(final_dict))
    encoded_loc = (encode_location(final_dict,len_longest))
    print 'encoded base and location'
    print len(encoded), len(encoded_bf), len(final_dict)
    bases = base_sequence_at_current_time_pr(encoded,encoded_bf)
    print 'encoded base seqs'
    #print len(bases[0][0])
    #bases = base_sequence_at_current_time_pr(encoded[1006],encoded_bf[1006])
    X = (structure_and_energy_at_current_time(bases,pid))
    #X2 = (structure_and_energy_at_current_time_with_location(bases,pid,final_dict,len_longest))
    print 'encoded strucs energy and locks'
    print len(X)
    # np.save(open(os.getcwd()+'/npsaves/X-exp-base-eli.npy','wb'),X2)
    # np.save(open(os.getcwd()+'/npsaves/X-exp-loc-eli.npy','wb'),X)
    # np.save(open(os.getcwd()+'/npsaves/y-exp-base-eli.npy','wb'),encoded_base)
    # np.save(open(os.getcwd()+'/npsaves/y-exp-loc-eli.npy','wb'),encoded_loc)

    #pickle.dump(X2,open(os.getcwd()+'/pickles/X-exp-base-'+str(pid),'wb'))
    if len(encoded) != 0:
        pickle.dump(X, open(os.getcwd()+'/pickles/X2-fast-loc-'+str(pid),'wb'))
        pickle.dump(encoded_base,open(os.getcwd()+'/pickles/y2-fast-base-'+str(pid),'wb'))
        pickle.dump(encoded_loc,open(os.getcwd()+'/pickles/y2-fast-loc-'+str(pid),'wb'))

for i in (content[50:]):
    speed(i)
