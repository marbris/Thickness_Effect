#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 12:08:38 2021

@author: martin
"""

import Cycler
from matplotlib import pyplot as plt

DataDir = 'Data/'

#%%

SL = Cycler.get_SampleList(DataDir)

#fig, ax= plt.subplots(figsize=(13,6))

Cap = []
Thick = []
AM_Mass = []

for Batch in list(SL):
    
    BI = Cycler.get_BatchInfo(Batch, DataDir)
    CC_Thickness = BI['CurrentCollector']['Thickness'] 
    CC_Mass = BI['CurrentCollector']['Mass'] 
    AM_frac = BI['Slurry']['AM']['Mass'] / (BI['Slurry']['AM']['Mass'] + BI['Slurry']['Binder']['Mass'] + BI['Slurry']['Carbon']['Mass'])
    
    
    for Sample in list(SL[Batch]):
        
        E_Thick = BI['Samples'][Sample]['ECCthickness'] - CC_Thickness
        E_Mass = BI['Samples'][Sample]['ECCmass'] - CC_Mass
        
        df = Cycler.get_CycleData(Batch, Sample, DataDir, Properties=['Cycle ID', 'Cap_DChg(mAh)'])
        
        df['Cap_DChg(mAh/um)'] = df['Cap_DChg(mAh)']/E_Thick
        
        
        Cap.append(df.iloc[5]['Cap_DChg(mAh)'])
        Thick.append(E_Thick)
        AM_Mass.append(E_Mass*AM_frac)
        
        #channel = BI['Samples'][Sample]['Cycler_Channel']
        #print(channel)
        #df.iloc[:-1].plot(x='Cycle ID', y='Cap_DChg(mAh)', marker='o', ax=ax, label = "L = {} $\mu m$, ({}  : {})".format(str(E_Thick),str(Batch),str(Sample)))



#ax.set_ylabel('Discharge capacity [mAh]', fontsize = 15)
#ax.set_xlabel('Cycle', fontsize = 15)



#%%

fig, ax= plt.subplots(figsize=(13,6))

ax.set_ylim((0, 5))
ax.set_xlim((0, 120))

d = 1.3 #cm
A = 3.14*(d/2)**2

X = Thick
Y = [Cap[i]/A for i in range(len(Cap))]

plt.plot(X, Y  ,marker='o', linestyle=' ', markerfacecolor='none', color = 'k')


ax.set_ylabel('Discharge capacity [mAh/cm$^2$]', fontsize = 15)
ax.set_xlabel('Thickness [$\mu m$]', fontsize = 15)

#%%

fig, ax= plt.subplots(figsize=(13,6))

#ax.set_ylim((0, 5))
#ax.set_xlim((0, 120))

d = 1.3 #cm
A = 3.14*(d/2)**2

X = AM_Mass
Y = [Cap[i]/AM_Mass[i] for i in range(len(Cap))]

plt.plot(X, Y  ,marker='o', linestyle=' ', markerfacecolor='none', color = 'k')


ax.set_ylabel('Discharge capacity [mAh/mg]', fontsize = 15)
ax.set_xlabel('AM mass [mg]', fontsize = 15)
#%%


SL = Cycler.get_SampleList(DataDir)


for Batch in list(SL):
    
    BI = Cycler.get_BatchInfo(Batch, DataDir)

    
    for Sample in list(SL[Batch]):

        channel = BI['Samples'][Sample]['Cycler_Channel']
        print(channel)
        
        
#%%


df = Cycler.get_CycleData('211123_NMC', 10, DataDir, Properties=['Cycle ID', 'Cap_DChg(mAh)'])