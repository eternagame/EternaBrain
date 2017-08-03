import numpy as np
from eterna_utils import get_pairmap_from_secstruct
import RNA
from subprocess import Popen, PIPE, STDOUT
import re
from difflib import SequenceMatcher
import copy

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

def find_parens(s):
    toret = {}
    pstack = []

    for i, c in enumerate(s):
        if c == '(':
            pstack.append(i)
        elif c == ')':
            if len(pstack) == 0:
                raise IndexError("No matching closing parens at: " + str(i))
            toret[pstack.pop()] = i

    if len(pstack) > 0:
        raise IndexError("No matching opening parens at: " + str(pstack.pop()))

    return toret

dot_bracket = '.....((((..((((....)))).)))).....'
seq_str = 'A'*len(dot_bracket)
def dsp(dot_bracket,seq_str):
    seq = list(seq_str)

    current_struc,_ = RNA.fold(seq_str)
    target_struc = encode_struc(dot_bracket)
    target_pm = get_pairmap_from_secstruct(dot_bracket)
    current_pm = get_pairmap_from_secstruct(current_struc)

    pairs = find_parens(dot_bracket)

    #print target_pm
    #print current_pm

    for base1,base2 in pairs.iteritems(): # corrects incorrect base pairings
        #print base1,base2
        if (seq[base1] == 'A' and seq[base2] == 'U') or (seq[base1] == 'U' and seq[base2] == 'A'):
            continue
        elif (seq[base1] == 'G' and seq[base2] == 'U') or (seq[base1] == 'U' and seq[base2] == 'G'):
            continue
        elif (seq[base1] == 'G' and seq[base2] == 'C') or (seq[base1] == 'C' and seq[base2] == 'G'):
            continue
        elif (seq[base1] == 'G' and seq[base2] == 'A'):
            seq[base1] = 'U'
        elif (seq[base1] == 'A' and seq[base2] == 'G'):
            seq[base1] = 'C'
        elif (seq[base1] == 'C' and seq[base2] == 'U'):
            seq[base1] = 'A'
        elif (seq[base1] == 'U' and seq[base2] == 'C'):
            seq[base1] = 'G'
        elif (seq[base1] == 'A' and seq[base2] == 'C'):
            seq[base1] = 'G'
        elif (seq[base1] == 'C' and seq[base2] == 'A'):
            seq[base1] = 'U'
        elif (seq[base1] == 'A' and seq[base2] == 'A'):
            seq[base1] = 'U'
        elif (seq[base1] == 'U' and seq[base2] == 'U'):
            seq[base1] = 'A'
        elif (seq[base1] == 'G' and seq[base2] == 'G'):
            seq[base1] = 'C'
        elif (seq[base1] == 'C' and seq[base2] == 'C'):
            seq[base1] = 'G'

    #print ''.join(seq)

    for i in range(len(target_pm)):
        if target_pm[i] == -1:
            seq[i] = 'A'
        else:
            continue

    for i in range(len(dot_bracket)):
        try:
            if dot_bracket[i] == '(':# or dot_bracket[i] == ')':
                #print dot_bracket[i]
                if dot_bracket[i-1] == '.' or dot_bracket[i-1] == ')' or dot_bracket[i+1] == '.' or dot_bracket[i+1] == ')':
                    #print i
                    if (seq[i] == 'G' and seq[target_pm[i]] == 'C') or (seq[i] == 'C' and seq[target_pm[i]] == 'G'):
                        continue
                    else:
                        seq[i] = 'G'
                        seq[target_pm[i]] = 'C'

                # elif dot_bracket[i+1] == '.' and dot_bracket[i+2] == '.' and dot_bracket[i+3] == '.' and dot_bracket[i+4] == '.':
                #     seq[i+1] = 'G'

            elif dot_bracket[i] == ')':# or dot_bracket[i] == ')':
                #print dot_bracket[i]
                if dot_bracket[i-1] == '.' or dot_bracket[i-1] == '(' or dot_bracket[i+1] == '.' or dot_bracket[i+1] == '(':
                    #print i
                    if (seq[i] == 'G' and seq[target_pm[i]] == 'C') or (seq[i] == 'C' and seq[target_pm[i]] == 'G'):
                        continue
                    else:
                        seq[i] = 'G'
                        seq[target_pm[i]] = 'C'

        except IndexError:
            continue

    for i in range(len(dot_bracket)):
        if dot_bracket[i] == '(':
            if dot_bracket[i+1] == '.' and dot_bracket[i+2] == '.' and dot_bracket[i+3] == '.' and dot_bracket[i+4] == '.':
                seq[i+1] = 'G'
            # elif (dot_bracket[i+1] == '.' and dot_bracket[i+2] == '('):
            #     seq[i+1] = 'G'

    for i in range(len(dot_bracket)):
        if dot_bracket[i] == '(' and dot_bracket[i+1] == '.' and dot_bracket[i+2] == '.' and dot_bracket[i+3] == '(': # UGUG superboost
            idx = target_pm[i]
            if dot_bracket[idx] == ')' and dot_bracket[idx-1] == '.' and dot_bracket[idx-2] == '.' and dot_bracket[idx-3] == ')':
                seq[i+1] = 'U'
                seq[i+2] = 'G'
                seq[idx-2] = 'U'
                seq[idx-1] = 'G'
            elif dot_bracket[idx] == ')' and dot_bracket[idx-1] == '.' and dot_bracket[idx-2] == ')':
                seq[i+1] = 'G'
                seq[idx-1] = 'G'

        elif dot_bracket[i] == '(' and dot_bracket[i+1] == '.' and dot_bracket[i+2] == '(': # G-G in 2 pair internal loop
            idx = target_pm[i]
            if dot_bracket[idx] == ')' and dot_bracket[idx-1] == '.' and dot_bracket[idx-2] == ')':
                seq[i+1] = 'G'
                seq[idx-1] = 'G'
            elif dot_bracket[idx] == ')' and dot_bracket[idx-1] == '.' and dot_bracket[idx-2] == '.' and dot_bracket[idx-3] == ')':
                seq[i+1] = 'G'
                seq[idx-1] = 'G'

    p = Popen(['../../../EteRNABot/eternabot/./RNAfold', '-T','37.0'], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
    pair = p.communicate(input=''.join(seq))[0]
    formatted = re.split('\s+| \(?\s?',pair)
    new_struc = formatted[1]
    new_pm = get_pairmap_from_secstruct(new_struc)
    match = SequenceMatcher(None,new_pm,target_pm).ratio()
    for j in range(3):
        for i in range(len(dot_bracket)):
            if new_pm == target_pm:
                print 'puzzle solved'
                break
            else:
                if new_pm[i] == target_pm[i]:
                    continue
                else:
                    paired = target_pm[i]
                    base1 = seq[i]
                    base2 = seq[paired]

                    if paired == -1: continue

                    seq[i] = base2
                    seq[paired] = base1
                    p = Popen(['../../../EteRNABot/eternabot/./RNAfold', '-T','37.0'], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
                    pair = p.communicate(input=''.join(seq))[0]
                    formatted = re.split('\s+| \(?\s?',pair)
                    new_pm = get_pairmap_from_secstruct(formatted[1])

                    new_match = SequenceMatcher(None,new_pm,target_pm).ratio()

                    if new_match > match:
                        match = copy.deepcopy(new_match)
                    else:
                        seq[i] = base1
                        seq[paired] = base2

                    

    for i in range(len(dot_bracket)):
        if new_pm[i] == target_pm[i]:
            continue
        else:
            paired = target_pm[i]
            seq[i] = 'G'
            seq[paired] = 'C'

    for j in range(3):
        for i in range(len(dot_bracket)):
            if new_pm == target_pm:
                print 'puzzle solved'
                break
            else:
                if new_pm[i] == target_pm[i]:
                    continue
                else:
                    paired = target_pm[i]
                    base1 = seq[i]
                    base2 = seq[paired]

                    if paired == -1: continue

                    seq[i] = base2
                    seq[paired] = base1
                    p = Popen(['../../../EteRNABot/eternabot/./RNAfold', '-T','37.0'], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
                    pair = p.communicate(input=''.join(seq))[0]
                    formatted = re.split('\s+| \(?\s?',pair)
                    new_pm = get_pairmap_from_secstruct(formatted[1])

                    new_match = SequenceMatcher(None,new_pm,target_pm).ratio()

                    if new_match > match:
                        match = copy.deepcopy(new_match)
                    else:
                        seq[i] = base1
                        seq[paired] = base2

    return ''.join(seq)

    cs,_ = RNA.fold(''.join(seq))
    current_pm = get_pairmap_from_secstruct(cs)

# print current_pm
# print target_pm
print dsp(dot_bracket,seq_str)
