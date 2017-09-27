import tensorflow as tf
import numpy as np
import copy
import RNA
from eterna_score import eternabot_score
from difflib import SequenceMatcher

def hot_one_state(seq,index,base):
    #array = np.zeros(NUM_STATES)
    copied_seq = copy.deepcopy(seq)
    copied_seq[index] = base
    return copied_seq

def convert(base_seq):
    str_struc = []
    for i in base_seq:
        if i == [1,0,0,0]:
            str_struc.append('A')
        elif i == [0,1,0,0]:
            str_struc.append('U')
        elif i == [0,0,1,0]:
            str_struc.append('G')
        elif i == [0,0,0,1]:
            str_struc.append('C')
    struc = ''.join(str_struc)
    s,_ = RNA.fold(struc)
    return_struc = []
    for i in s:
        if i == '.':
            return_struc.append(1)
        elif i == '(':
            return_struc.append(2)
        elif i == ')':
            return_struc.append(3)

    return np.array(return_struc)

def encode_struc(dots):
    s = []
    for i in dots:
        if i == '.':
            s.append(1)
        elif i == '(':
            s.append(2)
        elif i == ')':
            s.append(3)
    return s

def one_hot_seq(seq):
    onehot = []
    for base in seq:
        if base == 1:
            onehot.append([1.,0.,0.,0.])
        elif base == 2:
            onehot.append([0.,1.,0.,0.])
        elif base == 3:
            onehot.append([0.,0.,1.,0.])
        elif base == 4:
            onehot.append([0.,0.,0.,1.])

    return onehot

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def convert_to_list(base_seq):
    str_struc = []
    for i in base_seq:
        if i == 'A':
            str_struc.append([1,0,0,0])
        elif i == 'U':
            str_struc.append([0,1,0,0])
        elif i == 'G':
            str_struc.append([0,0,1,0])
        elif i == 'C':
            str_struc.append([0,0,0,1])
    #struc = ''.join(str_struc)
    return str_struc

def convert_to_str(base_str):
    str_struc = []
    for i in base_str:
        if i == [1,0,0,0]:
            str_struc.append('A')
        if i == [0,1,0,0]:
            str_struc.append('U')
        if i == [0,0,1,0]:
            str_struc.append('G')
        if i == [0,0,0,1]:
            str_struc.append('C')

    return ''.join(str_struc)

def convert_to_struc(base_seq):
    str_struc = []
    for i in base_seq:
        if i == [1,0,0,0]:
            str_struc.append('A')
        elif i == [0,1,0,0]:
            str_struc.append('U')
        elif i == [0,0,1,0]:
            str_struc.append('G')
        elif i == [0,0,0,1]:
            str_struc.append('C')
    struc = ''.join(str_struc)
    s,_ = RNA.fold(struc)
    return_struc = []
    for i in s:
        if i == '.':
            return_struc.append(1)
        elif i == '(':
            return_struc.append(2)
        elif i == ')':
            return_struc.append(3)

    return s

len_longest = 108

#current = convert(base_seq)
# dot_bracket = '....((((....(((.((.((....)))))))))))'
# target_struc = encode_struc(dot_bracket)
# cdb = '.....((((..(.((((...)))).)..))))....'
# current_struc = encode_struc(cdb)
# percent_match = similar(current_struc,target_struc)
# len_puzzle = len(target_struc)
# len_puzzle_float = len(target_struc) * 1.0

dot_bracket = '...((((((((((....))))......((((....))))))))))...'
target_struc = encode_struc(dot_bracket)
#cdb = '.((((....))))'
#current_struc = encode_struc(cdb)
#percent_match = similar(dot_bracket,cdb)
len_puzzle = len(target_struc)
len_puzzle_float = len(target_struc) * 1.0
# GAACGCACCUGCCUGUUUGGGGAGUAUGAA   GAACGCACCUGCCUGUUUGGGUAGCAUGAA   GAACGCACCUGCCUGUCUGGGUAGCAUGAA  GAACUCACCUGCCUGUCUUGGUAGCAUCAA
seq = 'GUAAGUAGUAUAAAAUGAGGACCAAGAGUAAAGGUAAAUACUAAAUGU'
current_seq = convert_to_list(seq)
cdb,_ = RNA.fold(seq)

NUM_STATES = len_puzzle #n_states
NUM_ACTIONS = 4
GAMMA = 0.5

session = tf.Session()
state = tf.placeholder("float", [None,NUM_STATES*4],name='state')
targets = tf.placeholder("float", [None, NUM_ACTIONS],name='targets')

hidden_weights1 = tf.get_variable('weights1',[NUM_STATES*4, 250],initializer=tf.truncated_normal_initializer())
hidden_weights2 = tf.get_variable('weights2',[250, 250],initializer=tf.truncated_normal_initializer())
out_weights = tf.get_variable('outw',[250, NUM_ACTIONS],initializer=tf.truncated_normal_initializer())

biases1 = tf.get_variable('biases1',[250],initializer=tf.truncated_normal_initializer())
biases2 = tf.get_variable('biases2',[250],initializer=tf.truncated_normal_initializer())
out_biases = tf.get_variable('outb',[NUM_ACTIONS],initializer=tf.truncated_normal_initializer())

l1 = tf.add(tf.matmul(state,hidden_weights1),biases1)
l1 = tf.nn.sigmoid(l1)

l2 = tf.add(tf.matmul(l1,hidden_weights2),biases2)
l2 = tf.nn.sigmoid(l2)

output = tf.add(tf.matmul(l2,out_weights),out_biases)

loss = tf.reduce_mean(tf.square(output - targets))
train_operation = tf.train.AdamOptimizer(0.1).minimize(loss)

session.run(tf.global_variables_initializer())

for i in range(2):
    state_batch = []
    rewards_batch = []
    rand_seq = []

    base_reward = similar(cdb,target_struc)

    # create a batch of states
    for state_index in range(len_puzzle):

        current_seq_shaped = np.array(current_seq).reshape([NUM_STATES*4])
        #print current_seq_shaped.shape
        #print np.array(current_seq).shape
        state_batch.append(current_seq_shaped)
        #print seq
        #print sec_struc

        # minus_action_index = (state_index - 1) % NUM_STATES # % NUM_STATES is so that it 'bounces' when it hits the end of the list
        # plus_action_index = (state_index + 1) % NUM_STATES
        # minus2_action_index = (state_index - 2) % NUM_STATES
        # plus2_action_index = (state_index + 2) % NUM_STATES

        A = [1,0,0,0]
        U = [0,1,0,0]
        G = [0,0,1,0]
        C = [0,0,0,1]

        # minus_action_state_reward = session.run(output, feed_dict={state: [hot_one_state(state_index,1)]})[0]
        # plus_action_state_reward = session.run(output, feed_dict={state: [hot_one_state(state_index,2)]})[0]
        # minus2_action_state_reward = session.run(output, feed_dict={state: [hot_one_state(state_index,3)]})[0]
        # plus2_action_state_reward = session.run(output, feed_dict={state: [hot_one_state(state_index,4)]})[0]
        # a_action_state_reward = session.run(output, feed_dict={state: [hot_one_state(seq,state_index,A)]})[0]
        # u_action_state_reward = session.run(output, feed_dict={state: [hot_one_state(seq,state_index,U)]})[0]
        # g_action_state_reward = session.run(output, feed_dict={state: [hot_one_state(seq,state_index,G)]})[0]
        # c_action_state_reward = session.run(output, feed_dict={state: [hot_one_state(seq,state_index,C)]})[0]

        a_change = hot_one_state(current_seq,state_index,A)
        u_change = hot_one_state(current_seq,state_index,U)
        g_change = hot_one_state(current_seq,state_index,G)
        c_change = hot_one_state(current_seq,state_index,C)

        # a_reward = eternabot_score(a_change)
        # u_reward = eternabot_score(u_change)
        # g_reward = eternabot_score(g_change)
        # c_reward = eternabot_score(c_change)

        a_struc = convert_to_struc(a_change)
        u_struc = convert_to_struc(u_change)
        g_struc = convert_to_struc(g_change)
        c_struc = convert_to_struc(c_change)

        # if a_struc.all() == sec_struc.all() or u_struc.all() == sec_struc.all() \
        #         or g_struc.all() == sec_struc.all() or c_struc.all() == sec_struc.all():
        #     a_reward,u_reward,g_reward,c_reward = 0,0,0,0
        # else:
        #     a_reward = (np.sum(a_struc == target_struc))/len_puzzle_float
        #     u_reward = (np.sum(u_struc == target_struc))/len_puzzle_float
        #     g_reward = (np.sum(g_struc == target_struc))/len_puzzle_float
        #     c_reward = (np.sum(c_struc == target_struc))/len_puzzle_float
        # print a_change,u_change,g_change,c_change

        # these action rewards are the results of the Q function for this state and the actions minus or plus
        # action_rewards = [states[minus_action_index] + GAMMA * np.max(minus_action_state_reward),
        #                   states[plus_action_index] + GAMMA * np.max(plus_action_state_reward),
        #                   states[minus2_action_index] + GAMMA * np.max(minus2_action_state_reward),
        #                   states[plus2_action_index] + GAMMA * np.max(plus2_action_state_reward)]

        a_reward = similar(a_struc,dot_bracket)
        u_reward = similar(u_struc,dot_bracket)
        g_reward = similar(g_struc,dot_bracket)
        c_reward = similar(c_struc,dot_bracket)

        action_rewards = [base_reward + GAMMA * a_reward,
                          base_reward + GAMMA * u_reward,
                          base_reward + GAMMA * g_reward,
                          base_reward + GAMMA * c_reward]

        # action_rewards = [a_reward,
        #                   u_reward,
        #                   g_reward,
        #                   c_reward]
        print action_rewards
        rewards_batch.append(action_rewards)

    session.run(train_operation, feed_dict={
        state: state_batch,
        targets: rewards_batch})

    print(([np.max(session.run(output, feed_dict={state: [np.array(current_seq).reshape([NUM_STATES*4])]}))
           for x in range(NUM_STATES)]))
    final_list = []


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
