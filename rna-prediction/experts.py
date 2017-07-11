import numpy as np
import os
from readData import experience
from encodeRNA import base_sequence_at_current_time_pr, structure_and_energy_at_current_time
from encodeRNA import encode_bases,encode_location,encode_movesets_style_pr
import pandas as pd
import ast

pidList = [6502996]
pid = 6502996
threshold = 50
len_longest = 108
filepath = os.getcwd() + '/movesets/moveset6-22a.txt'
print 'ready'

data,users,encoded_bf,lens = experience(pidList,threshold)
print 'experience'
encoded = encode_movesets_style_pr(data)
encoded_base = encode_bases(data)
encoded_loc = encode_location(data,len_longest)
print 'encoded'

# plist = []
# lens = []
# for pid in pidList:
#     puzzles_pid = (moveset_dataFrame.loc[moveset_dataFrame['pid'] == pid])
#     for uid in users:
#         puzzles_pid2 = puzzles_pid.loc[puzzles_pid['uid'] == uid]
#         p = (list(puzzles_pid2['move_set']))
#         plist.extend(p)
#     lens.append(len(list(puzzles_pid['move_set'])))
#
# bf_list = []
# for i in plist:
#  s1 = (ast.literal_eval(i))
#  s2 = s1['begin_from']
#  bf_list.append(s2)
#
# encoded_bf = []
# for start in bf_list:
#    enc = []
#    for i in start:
#        if i == 'A':
#            enc.append(1)
#        elif i == 'U':
#            enc.append(2)
#        elif i == 'G':
#            enc.append(3)
#        elif i == 'C':
#            enc.append(4)
#    encoded_bf.append(enc)
print 'encoded_bf'
print len(encoded), len(encoded_bf), len(data)
print lens
bases = base_sequence_at_current_time_pr(encoded,encoded_bf)

#bases = base_sequence_at_current_time_pr(encoded[1006],encoded_bf[1006])
X = (structure_and_energy_at_current_time(bases,pid))
