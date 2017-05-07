# -*- coding: utf-8 -*-
"""
Created on Thu Jan 05 18:30:26 2017

@author: Rohan
"""

#from selenium import webdriver
#from selenium.webdriver.common.keys import Keys
#import time
import RNA

def getData_pid(pid,pidList,movesets,structure): # returns moveset and puzzzle structure together
  i1 = pidList.index(pid)
  #return movesets[num]
  pid_structure = structure['pid']
  pid_puzzleList = list(pid_structure)
  i2 = pid_puzzleList.index(pid)

  return movesets[i1], structure['structure'][i2]


def getStructure(sequence):
    base_seq = []
    for i in sequence:
        if i == 1:
            base_seq.append('A')
        elif i == 2:
            base_seq.append('U')
        elif i == 3:
            base_seq.append('G')
        elif i == 4:
            base_seq.append('C')

    base_seq = ''.join(base_seq)
    struc,energy = RNA.fold(base_seq)
    return struc,energy


'''
def getStructure_0(sequence): # gets structure using Vienna algorithm # DEPRECATED
    driver = webdriver.Chrome('/Users/rohankoodli/Desktop/chromedriver')
    driver.get("http://rna.tbi.univie.ac.at//cgi-bin/RNAWebSuite/RNAfold.cgi")
    inputElement = driver.find_element_by_id("SCREEN")
    for i in sequence:
        if i == 1:
            inputElement.send_keys('A')
        elif i == 2:
            inputElement.send_keys('U')
        elif i == 3:
            inputElement.send_keys('G')
        elif i == 4:
            inputElement.send_keys('C')

    driver.find_element_by_class_name('proceed').click()
    time.sleep(20)
    web_struc = driver.find_element_by_id('MFE_structure_span').text

    struc = []
    for i in web_struc:
        if i == '.' or i == '(' or i == ')':
            struc.append(i)
    struc = ''.join(struc)
    return struc

'''
#driver = webdriver.Chrome('/Users/rohankoodli/Desktop/chromedriver')
#driver.get("http://rna.tbi.univie.ac.at//cgi-bin/RNAWebSuite/RNAfold.cgi")
#inputElement = driver.find_element_by_id("SCREEN")
#seq = ['G','G','G','A','A','A','C','C','C']
#for i in seq:
#    inputElement.send_keys(i)
#driver.find_element_by_class_name('proceed').click()
#
#time.sleep(20)
#web_struc = driver.find_element_by_id('MFE_structure_span').text
#
#print web_struc
#
#
#struc = []
#for i in (web_struc):
#    if i == '.' or i == '(' or i == ')':
#        struc.append(i)
#
#struc = ''.join(struc)
#print struc
#print type(struc)


#
#from Bio import SeqIO
#sequences = ['AAA']  # add code here
#with open("example.fasta", "w") as output_handle:
#    SeqIO.write(sequences, output_handle, "fasta")


abc = [[[1,1,1,1],[000000],[010010],[-1.62]],[[1,2,3,4],[000000],[101000],[2.34]]]
xyz = [[[1,2],[0,1],[0,1],[-1.62,0]],[[1,2],[1,0],[0,1],[-1.62,0]]]
xyz = [[1,2],[0,1],[0,1],[-1.62,0]]
dfg = [[4,2],[3,7]]
X = [[1,2,3,4],[3,2,3,2]]
Y = [[4,2],[3,3]]
x = [1,2,3,4]
y = [5,6,7,8]
'''
import tflearn
from sklearn import tree

tflearn.init_graph(num_cores=1)

net = tflearn.input_data(shape=[None,2,4])
net = tflearn.fully_connected(net, 64)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 10, activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

model = tflearn.DNN(net)
model.fit(X,Y)
'''
