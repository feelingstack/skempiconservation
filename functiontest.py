# -*- coding:utf-8 -*-
import pandas as pd
import os
import numpy as np
'''
pf = pd.read_excel(r'../../excell/450_3.xlsx')
list1 = list(pf.loc[pf.index[:], 'chain'])
print(list1)
print(type(list1))
'''
n = np.array([[1, 2], [3, 4]])
dict = {}
dict['1a22'] = n
list1 = []
list1.append(dict['1a22'][1][1])
print(list1)
'''

'''