import tensorflow as tf
import numpy as np
import copy
from multiprocessing import Pool, Process
from contextlib import closing
import concurrent.futures
import RNA
from eterna_score import eternabot_score

def hot_one_state(seq,index,base):
    #array = np.zeros(NUM_STATES)
    copied_seq = copy.deepcopy(seq)
    copied_seq[index] = base
    return copied_seq

# def convert(base_seq):
#     str_struc = []
#     for i in base_seq:
#         if i == [1,0,0,0]:
#             str_struc.append('A')
#         elif i == [0,1,0,0]:
#             str_struc.append('U')
#         elif i == [0,0,1,0]:
#             str_struc.append('G')
#         elif i == [0,0,0,1]:
#             str_struc.append('C')
#     struc = ''.join(str_struc)
#     s,_ = RNA.fold(struc)
#     return_struc = []
#     for i in s:
#         if i == '.':
#             return_struc.append(1)
#         elif i == '(':
#             return_struc.append(2)
#         elif i == ')':
#             return_struc.append(3)
#
#     return np.array(return_struc)

len_longest = 108

base_seq = [1,1,1,1,1,1,1,1,1,1,1,1,1]
#current = convert(base_seq)
target_struc = np.array([2,2,2,1,1,1,3,3,3])
len_puzzle = len(target_struc)
len_puzzle_float = len(target_struc) * 1.0

NUM_STATES = len_puzzle#n_states
NUM_ACTIONS = 4
GAMMA = 0.5

session = tf.Session()
state = tf.placeholder("float", [None, NUM_STATES])
targets = tf.placeholder("float", [None, NUM_ACTIONS])

hidden_weights = tf.Variable(tf.constant(0., shape=[NUM_STATES, NUM_ACTIONS]))

output = tf.matmul(state, hidden_weights)

loss = tf.reduce_mean(tf.square(output - targets))
train_operation = tf.train.AdamOptimizer(0.1).minimize(loss)

session.run(tf.global_variables_initializer())

for i in range(3):
    state_batch = []
    rewards_batch = []
    rand_seq = []
    for i in range(len_puzzle):
        seq = np.zeros(4)
        seq[0] = 1
        np.random.shuffle(seq)
        rand_seq.append(seq)
    rand_seq_flat = np.array(rand_seq).reshape([-1,4*len_puzzle])
    base_reward = eternabot_score(rand_seq)['finalscore'] / 100.0

    # create a batch of states
    for state_index in range(len_puzzle):

        state_batch.append(rand_seq_flat)
        #print seq
        #print sec_struc

        # minus_action_index = (state_index - 1) % NUM_STATES # % NUM_STATES is so that it 'bounces' when it hits the end of the list
        # plus_action_index = (state_index + 1) % NUM_STATES
        # minus2_action_index = (state_index - 2) % NUM_STATES
        # plus2_action_index = (state_index + 2) % NUM_STATES

        A = np.array([1.,0.,0.,0.])
        U = np.array([0.,1.,0.,0.])
        G = np.array([0.,0.,1.,0.])
        C = np.array([0.,0.,0.,1.])

        # minus_action_state_reward = session.run(output, feed_dict={state: [hot_one_state(state_index,1)]})[0]
        # plus_action_state_reward = session.run(output, feed_dict={state: [hot_one_state(state_index,2)]})[0]
        # minus2_action_state_reward = session.run(output, feed_dict={state: [hot_one_state(state_index,3)]})[0]
        # plus2_action_state_reward = session.run(output, feed_dict={state: [hot_one_state(state_index,4)]})[0]
        # a_action_state_reward = session.run(output, feed_dict={state: [hot_one_state(seq,state_index,A)]})[0]
        # u_action_state_reward = session.run(output, feed_dict={state: [hot_one_state(seq,state_index,U)]})[0]
        # g_action_state_reward = session.run(output, feed_dict={state: [hot_one_state(seq,state_index,G)]})[0]
        # c_action_state_reward = session.run(output, feed_dict={state: [hot_one_state(seq,state_index,C)]})[0]

        a_change = hot_one_state(rand_seq,state_index,A)
        u_change = hot_one_state(rand_seq,state_index,U)
        g_change = hot_one_state(rand_seq,state_index,G)
        c_change = hot_one_state(rand_seq,state_index,C)

        a_reward = eternabot_score(a_change)['finalscore'] / 100.0

        # a_struc = convert(a_change)
        # u_struc = convert(u_change)
        # g_struc = convert(g_change)
        # c_struc = convert(c_change)

        # if a_struc.all() == sec_struc.all() or u_struc.all() == sec_struc.all() \
        #         or g_struc.all() == sec_struc.all() or c_struc.all() == sec_struc.all():
        #     a_reward,u_reward,g_reward,c_reward = 0,0,0,0
        # else:
        #     a_reward = (np.sum(a_struc == target_struc))/len_puzzle_float
        #     u_reward = (np.sum(u_struc == target_struc))/len_puzzle_float
        #     g_reward = (np.sum(g_struc == target_struc))/len_puzzle_float
        #     c_reward = (np.sum(c_struc == target_struc))/len_puzzle_float
        #print a_change,u_change,g_change,c_change

        # these action rewards are the results of the Q function for this state and the actions minus or plus
        # action_rewards = [states[minus_action_index] + GAMMA * np.max(minus_action_state_reward),
        #                   states[plus_action_index] + GAMMA * np.max(plus_action_state_reward),
        #                   states[minus2_action_index] + GAMMA * np.max(minus2_action_state_reward),
        #                   states[plus2_action_index] + GAMMA * np.max(plus2_action_state_reward)]

        action_rewards = [base_reward + GAMMA * a_reward,
                          base_reward + GAMMA * u_reward,
                          base_reward + GAMMA * g_reward,
                          base_reward + GAMMA * c_reward]

        # action_rewards = [a_reward,
        #                   u_reward,
        #                   g_reward,
        #                   c_reward]
        #print action_rewards
        rewards_batch.append(action_rewards)

    session.run(train_operation, feed_dict={
        state: state_batch,
        targets: rewards_batch})

    # print(([target_struc[x] + np.max(session.run(output, feed_dict={state: [np.random.randint(1,5,size=(len_puzzle))]}))
    #        for x in range(NUM_STATES)]))
    final_list = []
    for x in range(NUM_STATES):
        #print 'iteration',x
        final_list.append(np.max(session.run(output, feed_dict={state: [np.random.randint(1,5,size=(len_puzzle))]})))
    print final_list

# def run(_):
#     return q()
# if __name__ == '__main__':
#     with closing(Pool(processes=2)) as p:
#         print p.map(run,range(1))
#         p.terminate()

#q()
# with concurrent.futures.ProcessPoolExecutor() as executor:
#     nums = [10,10,10]
#     for i,j in zip(nums,executor.map(q,nums)):
#         print '\n'
#         print 'State %i completed' % (nums.index(i))
#         print 'Number of states: %d' % i

# nums = [10,10,10]
# for i in nums:
#     q(1)
