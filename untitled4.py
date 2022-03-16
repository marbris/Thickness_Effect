#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 18:15:51 2022

@author: martin
"""

import Cycler
from matplotlib import pyplot as plt


DataDir = r'Data/'

SL = Cycler.get_SampleList(DataDir)

BatchLabel = '220203_NMC'

#%%



#%%

fig, ax= plt.subplots(figsize=(13,6))

ax2 = ax.twinx()

ax.set_ylim((1.5, 8))
#ax.set_xlim((1e-1, 2e2))

SampleID = '20'
df = Cycler.get_SampleData(BatchLabel, '20', DataDir)

SI, BI = Cycler.get_SampleInfo(BatchLabel, '20', DataDir)

NC = len(df['Cycles'])

for ci in range(NC):
    NS = len(df['Cycles'][ci]['Steps'])
    
    for si in range(NS):
        V = df['Cycles'][ci]['Steps'][si]['Voltage(V)']
        I = df['Cycles'][ci]['Steps'][si]['Current(A)']
        t = df['Cycles'][ci]['Steps'][si]['Test_Time(s)']
        
        
        ax.plot(t,V, marker = '.', color = 'k')
        ax2.plot(t,I, marker = '.', color = 'r')



ax.set_xlabel('test time [s]', fontsize = 15)
ax2.set_ylabel('Current [A]', fontsize = 15)
ax.set_ylabel('Voltage [V]', fontsize = 15)

    