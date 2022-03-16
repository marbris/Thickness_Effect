#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:37:15 2022

@author: martin
"""

import dfall_dfCrit
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%% Calculating volumetric capacity




dfall, dfCrit = dfall_dfCrit.dfall_dfCrit()


df = dfall.loc[dfall['C-rate(rnd)'] == 0.1, ('Batch', 'Cycle_ID', 'Cathode', 'Capacity(mAh/cm2)', 'Thickness(um)')].copy()

#area capacity/thickness
df.loc[:,'Capacity(mAh/cm3)'] = df.loc[:,'Capacity(mAh/cm2)']/(df.loc[:,'Thickness(um)']*1e-4)

df.loc[df['Batch']!='211202_NMC', 'Cycle_ID'] = -1

groups = df.groupby(by=['Batch', 'Cycle_ID'])


#%% Penetration depth vs Crate double column figure

Cols = ['#e36414','#0db39e','#748cab','#8a5a44']
markers = ['o','s','d', 'v']
lines = ['-',':','--','']
def stylefun(G):
    return (Cols[G], markers[G], lines[G])

fig, ax= plt.subplots(figsize=(8,6))    



#ax.set_ylim((ymin, ymax))
#ax.set_xlim((xmin, xmax))


kkcms = np.array([2**i for i in range(-1,8,2)])*1e-8
kkmuh = kkcms*3.6e11


Cc = dfCrit.loc[dfCrit['Batch']=='211202_NMC', 'C-rate(rnd)']
index2 = (Cc > 0.1) & (Cc < 4.8) 
ms = 4
lw = 1.5
for i, (name, group) in enumerate(dfCrit.groupby(by=['Batch'])):
    
    df = group
    
    
    index = ~df['Thickness_lo(um)'].isnull() & ~df['Thickness_hi(um)'].isnull()
    
    if name == '211202_NMC': 
        index = index & index2
        label = 'Current Work, NMC'
    else:
        label = name
    
    
    cc = np.array(df.loc[index,'C-rate_mean(1/h)'].tolist())
    tt = np.array(df.loc[index,'Thickness_max(um)'].tolist())
    
    qa = np.array([])
    for it in tt:
        temp = dfall.loc[(dfall['Batch']==name) & (dfall['Thickness(um)']==it) & (dfall['C-rate(rnd)']==0.1), 'Avg_DCapacity(mAh/cm2)'].mean()
        #if len(temp)>1:
        #    print(name + ' ' + str(it) + ' ' + str(temp))
        qa = np.append(qa, temp)
    
    qv = qa/(tt*1e-4)
    
    rl2 = cc*tt**2/3.6e11
    
    ax.plot(qv, rl2*1e8, 
            marker = stylefun(i)[1], 
            linestyle = stylefun(i)[2], 
            color = stylefun(i)[0],  
            label = label, 
            markersize=ms*2, 
            linewidth = lw)

ax.set_ylabel('RL$^2_d$ [10$^{-8}$ cm$^2$/s]')
ax.set_xlabel('volumetric capacity, Q$_v$ [mAh/cm3]')



handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, 
          ncol = 1, 
          framealpha = 0, 
          fontsize = 15,
          loc = 'best',
          columnspacing=0.5, 
          handletextpad = 0.3, 
          labelspacing = 0.3)



fig.tight_layout()
plt.show()


