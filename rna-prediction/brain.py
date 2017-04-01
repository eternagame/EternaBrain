# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 21:49:27 2017

@author: rohankoodli
"""

import numpy as np
import os
from readData import read_movesets_all, read_movesets_pid, read_structure
from encodeRNA import encode_movesets, encode_structure

structurepath = os.getcwd() + '/movesets/puzzle-structure-data.txt'
movesetpath = os.getcwd() + '/movesets/move-set-11-14-2016.txt'

data, users = read_movesets_pid(movesetpath,6892348)
encoded = np.matrix(encode_movesets(data))

structures = read_structure(structurepath)
structures_encoded = encode_structure(structures['structure'][19684])

