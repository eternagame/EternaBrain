import numpy as np
import os
from readData import experience, experience_labs, read_movesets_uid_pid, read_movesets_uid
from encodeRNA import base_sequence_at_current_time_pr, structure_and_energy_at_current_time
from encodeRNA import encode_bases,encode_location,encode_movesets_style_pr
from encodeRNA import structure_and_energy_at_current_time_with_location
import ast
import concurrent.futures
import time
import pickle
from getData import getPid

with open(os.getcwd()+'/movesets/teaching-puzzle-ids.txt') as f:
    progression = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
progression = [x.strip() for x in progression]
progression = [int(x) for x in progression]
progression.extend([6502966,6502968,6502973,6502976,6502984,6502985,6502993,
                6502994,6502995,6502996,6502997,6502998,6502999,6503000])

content = progression
uidList = [36921]
#uidList = experience(3000) # top 99 percentile
print len(uidList)
#print content
len_longest = 400

def prep(pid):
    """
    Selects expert move sets and saves in npy file

    :param pid: Puzzle ID
    :return: npy saved files of training data
    """
    # pidList = pid
    # pid = pidList[0]
    threshold = 50

    print 'ready with pid %i' % pid

    data,users,encoded_bf,lens = experience_labs(pid, threshold)
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

def read(pid,uidList):
    """
    Returns training data for expert players of one puzzle

    :param pid: Puzzle ID
    :param uidList: List of user IDs
    :return: Pickled training data
    """

    print 'ready with pid %i' % pid

    #uidList.remove(87216)
    #uidList = [8627]
    print uidList
    final_dict = []
    bf_list = []
    #start = time.time()
    for user in uidList:
        print user
        data = read_movesets_uid_pid(user,pid)
        #data = read_movesets_uid(user)
        print 'data read'
        if not data:
            print 'user %i with pid %i list empty' % (user,pid)
            continue
        else:
            for i in data:
                print 'formatting into list'
                s1 = ast.literal_eval(i)
                s2 = s1['moves']
                s3 = s1['begin_from']
                final_dict.append(s2)
                bf_list.append(s3)
                print 'done formatting list'
            print 'user %i done with pid %i' % (user,pid)
    #print time.time() - start()
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

    encoded = encode_movesets_style_pr(final_dict)
    encoded_base = (encode_bases(final_dict))
    encoded_loc = (encode_location(final_dict,len_longest))
    print 'encoded base and location'
    print len(encoded), len(encoded_bf), len(final_dict)
    bases = base_sequence_at_current_time_pr(encoded,encoded_bf)
    print 'encoded base seqs'
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
    pickle.dump(X, open(os.getcwd()+'/pickles/X-hog-loc-'+str(pid),'wb'))
    pickle.dump(encoded_base,open(os.getcwd()+'/pickles/y-hog-base-'+str(pid),'wb'))
    pickle.dump(encoded_loc,open(os.getcwd()+'/pickles/y-hog-loc-'+str(pid),'wb'))

def read2(data,pids):
    """
    Returns CNN-ready training data for expert players of several
    puzzles

    :param data: List of move set data
    :param pids: List of puzzle IDs
    :return: Pickled training data
    """
    #print 'ready with pid %i' % pid

    #uidList.remove(87216)
    #uidList = [8627]
    #print uidList
    final_dict = []
    bf_list = []
    #start = time.time()
    for pid,sol in zip(pids,data):
        print pid
        #data = read_movesets_uid(user)
        print 'data read'
        if not data:
            print 'empty'
            continue
        else:
            print 'formatting into list'
            s1 = ast.literal_eval(sol)
            s2 = s1['moves']
            s3 = s1['begin_from']
            final_dict.append(s2)
            bf_list.append(s3)
            print 'done formatting list'
            #print 'user %i done with pid %i' % (user,pid)
    #print time.time() - start()
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

        encoded = encode_movesets_style_pr(final_dict)
        encoded_base = (encode_bases(final_dict))
        encoded_loc = (encode_location(final_dict,len_longest))
        print 'encoded base and location'
        print len(encoded), len(encoded_bf), len(final_dict)
        bases = base_sequence_at_current_time_pr(encoded,encoded_bf)
        print 'encoded base seqs'
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
        pickle.dump(X, open(os.getcwd()+'/pickles/X-hog-loc-'+str(pid),'wb'))
        pickle.dump(encoded_base,open(os.getcwd()+'/pickles/y-hog-base-'+str(pid),'wb'))
        pickle.dump(encoded_loc,open(os.getcwd()+'/pickles/y-hog-loc-'+str(pid),'wb'))

def run(_):
    """
    Meant for running/parallelizing training data preparation
    :param _: Not used
    :return: Runs prep() function
    """
    return prep()

# for i in reversed(content[:content.index(7165340)]):
#     read(i,uidList)

for i in range(len(content)/2):
    read(content[i],uidList)

# data, pids = read_movesets_uid(36921)
# read2(data,pids)
# with concurrent.futures.ProcessPoolExecutor() as executor:
#     #x = [6502996,6502990]
#     for i,j in zip(x,executor.map(read,content[80:])):
#         print 'running'

#prep()
# if __name__ == '__main__':
#     with closing(Pool(processes=8)) as p: # 8 cores
#         print p.map(run,range(1))
#         p.terminate()
