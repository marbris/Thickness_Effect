#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 13:39:20 2021

@author: martin
"""
#%%
import os
import re
import simplejson as json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import Cycler



def list_files(dir, RegExp):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            if re.search(RegExp, name):
                r.append(os.path.join(root, name))
    return r


#%%


filename = 'Data/Data_220203_NMC/BET_220203_NMC.json'

with open(filename) as file:
    BET_Data=json.load(file)




#%%


df = Cycler.get_CycleData('211117_NMC_1', 1, 'Data', Properties=['Cycle ID', 'Cap_DChg(mAh)'])

#%%


regexp=r"Batch_.*\.json$"

r = list_files(r'Data/', regexp)

BatchName = r[0]

with open(BatchName) as file:
    BatchData=json.load(file)
        

#%%

fig, ax= plt.subplots(figsize=(13,6))

regexp=r"CoinCell_.*\.json$"

r = list_files(r'Data/', regexp)

#CoinCellName = r[1]
for CoinCell in r:
    
    directory, filename = os.path.split(CoinCell)
    CellIndex, BatchLabel = re.match('CoinCell_(\d+)_(.*)\.json', filename).groups()
    
    Batch = os.path.join(directory, 'Batch_{}.json'.format(BatchLabel))
    
    with open(Batch) as file:
        BatchData=json.load(file)
    
    
    
    with open(CoinCell) as file:
        CellData=json.load(file)
    
    N = len(CellData['Cycles'])-1
    Q = np.array([CellData['Cycles'][i]['Cap_DChg(mAh)'] for i in range(N)])
    Ci = np.array([CellData['Cycles'][i]['Cycle ID'] for i in range(N)])

    plt.plot(Ci, Q, 'o')



#%%

CoinCellName = 'Data/211123_NMC/CoinCell_12_211123_NMC.json'

regexp='.*CoinCell_(\d+)_(.*)\.json'

CellIndex, BatchLabel = re.match(regexp, CoinCellName).groups()

directory, filename = os.path.split(CoinCellName)

#%%

filename = 'Data/211117_NMC_1/Batch_211117_NMC_1.json'

with open(filename) as file:
        BatchData=json.load(file)
        

#%%



#def get_PointData(BatchLabel, SampleID, DataDirectory, Properties=[]):   
    
"""
Step_df = get_StepData(BatchLabel, SampleID, DataDirectory, Properties=[])
    
This function extracts the step-level data from [DataDirectory]/Data_[BatchLabel]/CoinCell_[SampleID]_[BatchLabel].json
and returns it as a pandas dataframe.

The Properties argument specifies what properties should be extracted. The default is to extract all properties.

"""

SampleData = Cycler.get_SampleData('220209_NMC', '14', 'Data/')
Properties=[]

#Here I'm collecting all the keys under 'Steps'
AllKeys = list(SampleData['Cycles'][0]['Steps'][0])

#Here I'm selecting the keys that are lists.
PointDataKeys = []
for key in AllKeys:
    if isinstance(SampleData['Cycles'][0]['Steps'][0][key], list):
        PointDataKeys.append(key)



if Properties == []:
    Properties = PointDataKeys
else:
    nonkeys=[]
    for key in Properties:
        if key not in PointDataKeys:
            nonkeys.append(key)
    if nonkeys:
        raise KeyError('{0} Do not exist. Try these: {1}'.format(nonkeys, PointDataKeys))



#starting a list of lists to make a pandas data frame later.
point_df = pd.DataFrame(data=[], columns = Properties)
NCyc = len(SampleData['Cycles'])
for Cyc_i in range(NCyc):
    NStep = len(SampleData['Cycles'][Cyc_i]['Steps'])
    for Step_i in range(NStep):
        temp_df = pd.DataFrame()
        for key in Properties:
            temp_df[key] = SampleData['Cycles'][Cyc_i]['Steps'][Step_i][key]
             
        point_df = point_df.append(temp_df, ignore_index = True)



#point_df = pd.DataFrame(data, columns = Properties)

#return Step_df

#%%

fig, ax= plt.subplots(figsize=(13,6))

df = Cycler.get_PointData('220209_NMC', '14', 'Data/', Properties = ['Test_Time(s)', 'Current(A)','Cycle_Index'])

df.set_index('Test_Time(s)', inplace=True)
df.groupby('Cycle_Index')['Current(A)'].plot(ax=ax, legend=True)

#%%


