import numpy as np
import os
from readData import experience_labs, experience, read_movesets_uid_pid
from encodeRNA import base_sequence_at_current_time_pr, structure_and_energy_at_current_time
from encodeRNA import encode_bases,encode_location,encode_movesets_style_pr
from encodeRNA import structure_and_energy_at_current_time_with_location
#import pandas as pd
import ast
import concurrent.futures

with open(os.getcwd()+'/movesets/teaching-puzzle-ids.txt') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]
content = [int(x) for x in content]
progression = [6502966,6502968,6502973,6502976,6502984,6502985,6502993, \
                6502957,6502994,6502995,6502996,6502997,6502998,6502999,6503000]
content.extend(progression)
print content
len_longest = 108

def prep(pid):
    # pidList = pid
    # pid = pidList[0]
    threshold = 50

    print 'ready with pid %i' % pid

    data,users,encoded_bf,lens = experience_labs(pid,threshold)
    print 'experience_labs'
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
    X = np.array(structure_and_energy_at_current_time(bases,pid,data,len_longest))
    np.save(open(os.getcwd()+'/npsaves/X-exp-'+str(pid),'wb'),X)
    np.save(open(os.getcwd()+'/npsaves/y-exp-base-'+str(pid),'wb'),encoded_base)
    np.save(open(os.getcwd()+'/npsaves/y-exp-loc-'+str(pid),'wb'),encoded_loc)

def read(pid):
    print 'ready with pid %i' % pid
    #uidList = [267,8627,57675,124111,11775,24263,40299,42833,52361,32627,219533,196596,55836,231387]
    uidList = experience(10000)
    print uidList
    final_dict = []
    bf_list = []
    for user in uidList:
        data = read_movesets_uid_pid(user,pid)
        for i in data:
            s1 = ast.literal_eval(i)
            s2 = s1['moves']
            s3 = s1['begin_from']
            final_dict.append(s2)
            bf_list.append(s3)
    print "data read"
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

    encoded = encode_movesets_style_pr(final_dict)
    encoded_base = np.array(encode_bases(final_dict))
    encoded_loc = np.array(encode_location(final_dict,len_longest))
    print 'encoded base and location'
    print len(encoded), len(encoded_bf), len(final_dict)
    bases = base_sequence_at_current_time_pr(encoded,encoded_bf)
    print 'encoded base seqs'
    #bases = base_sequence_at_current_time_pr(encoded[1006],encoded_bf[1006])
    X = np.array(structure_and_energy_at_current_time(bases,pid))
    X2 = np.array(structure_and_energy_at_current_time_with_location(bases,pid,final_dict,len_longest))
    print 'encoded strucs energy and locks'
    print len(X)
    np.save(open(os.getcwd()+'/npsaves/X-exp-base-'+str(pid),'wb'),X2)
    np.save(open(os.getcwd()+'/npsaves/X-exp-loc-'+str(pid),'wb'),X)
    np.save(open(os.getcwd()+'/npsaves/y-exp-base-'+str(pid),'wb'),encoded_base)
    np.save(open(os.getcwd()+'/npsaves/y-exp-loc-'+str(pid),'wb'),encoded_loc)

def run(_):
    return prep()

with concurrent.futures.ProcessPoolExecutor() as executor:
    #x = [6502996,6502990]
    for i,j in zip(x,executor.map(read,content[80:])):
        print 'running'

#prep()
# if __name__ == '__main__':
#     with closing(Pool(processes=8)) as p: # 8 cores
#         print p.map(run,range(1))
#         p.terminate()
