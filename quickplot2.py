#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 21:22:26 2022

@author: martin
"""

import Cycler
import matplotlib.pyplot as plt
import simplejson as json
import numpy as np

#%%

SampleList = { '220203_NMC': ['03', '04', '05', '06', '09', '17', '18', '21', '22', '23']}
SampleList = { '220203_NMC': ['03', '04', '06', '18', '21', '22', '23']}

df = Cycler.get_BatchCycles(SampleList)


#filename = 'Data/Data_220203_NMC/Batch_220203_NMC.json'



#with open(filename) as file:
#        BatchInfo=json.load(file)



fig, ax= plt.subplots(figsize=(13,6))


for name, group in df.groupby('SampleID'):
    group.plot(x='Cycle_ID', y='Capacity(mAh/cm2)', ax=ax, label = name)


#%%

linestyles=['-',':','--','-.']

fig, ax= plt.subplots(figsize=(10,5))    
    

Batch = '220203_NMC'

for i, sample in enumerate(SampleList[Batch]):
    SI, BI = Cycler.get_SampleInfo(Batch, sample)
    thickness = SI['ECCthickness'] - BI['CurrentCollector']['Thickness']
    dft = Cycler.get_PointData(Batch, sample)
    dft.plot(x = 'Test_Time(s)', y = 'Voltage(V)', ax=ax, label = '{} $\mu m$, ID: {}'.format(thickness, sample), linestyle = linestyles[np.mod(i,4)])


plt.xlabel('Time (s)', fontsize = 18)
plt.ylabel('Voltage (V)', fontsize = 18)


handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, ncol = 4, loc='lower center', framealpha = 0, fontsize=15)

plt.show()


#%%

linestyles=['-',':','--','-.']

fig, ax= plt.subplots(figsize=(13,6))    
    

Batch = '220203_NMC'

for i, sample in enumerate(SampleList[Batch]):
    SI, BI = Cycler.get_SampleInfo(Batch, sample)
    thickness = SI['ECCthickness'] - BI['CurrentCollector']['Thickness']
    dft = Cycler.get_PointData(Batch, sample)
    dft.plot(x = 'Test_Time(s)', y = 'Current(A)', ax=ax, label = '{} $\mu m$, ID: {}'.format(thickness, sample), linestyle = linestyles[np.mod(i,4)])


plt.xlabel('Time (s)', fontsize = 18)
plt.ylabel('Current(A)', fontsize = 18)


handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, ncol = 4, loc='lower center', framealpha = 0)

plt.show()
