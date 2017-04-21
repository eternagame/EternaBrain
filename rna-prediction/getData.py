# -*- coding: utf-8 -*-
"""
Created on Thu Jan 05 18:30:26 2017

@author: Rohan
"""

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

def getData_pid(pid,pidList,movesets,structure): # returns moveset and puzzzle structure together
  i1 = pidList.index(pid)
  #return movesets[num]
  pid_structure = structure['pid']
  pid_puzzleList = list(pid_structure)
  i2 = pid_puzzleList.index(pid)
  
  return movesets[i1], structure['structure'][i2]

def getStructure(sequence): # gets structure using Vienna algorithm
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


driver = webdriver.Chrome('/Users/rohankoodli/Desktop/chromedriver')
driver.get("http://rna.tbi.univie.ac.at//cgi-bin/RNAWebSuite/RNAfold.cgi")
inputElement = driver.find_element_by_id("SCREEN")
seq = ['G','G','G','A','A','A','C','C','C']
for i in seq:
    inputElement.send_keys(i)
driver.find_element_by_class_name('proceed').click()
time.sleep(20)
web_struc = driver.find_element_by_id('MFE_structure_span').text
print web_struc


struc = []
for i in (web_struc):
    if i == '.' or i == '(' or i == ')':
        struc.append(i)

struc = ''.join(struc)
print struc
print type(struc)

