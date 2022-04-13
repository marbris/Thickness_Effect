#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:37:15 2022

@author: martin
"""
#%%
import main
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 15

Axes = {'titlesize': SMALL_SIZE,    # fontsize of the axes title
        'labelsize': MEDIUM_SIZE}   # fontsize of the x and y labels
plt.rc('axes', **Axes)


plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

plt.rc('figure', dpi = 200)

Col1= 8.9/2.54 # 8.9cm single column figure width in Nature
Col2= 18.3/2.54 # 18.3cm double column figure width in Nature

AR = 1/1.618

dfall, dfCapCrit, dfOPCrit = main.dfall_dfCrit()

#%% Calculating volumetric capacity








df = dfall.loc[dfall['C-rate(1/h)'] == 0.1, ('Batch', 'Cycle', 'Cathode', 'Discharge_Capacity(mAh/cm2)', 'Thickness(um)')].copy()

#area capacity/thickness
df.loc[:,'Capacity(mAh/cm3)'] = df.loc[:,'Discharge_Capacity(mAh/cm2)']/(df.loc[:,'Thickness(um)']*1e-4)

df.loc[df['Batch']!='211202_NMC', 'Cycle'] = -1

groups = df.groupby(by=['Batch', 'Cycle'])


#%% Penetration depth vs Crate double column figure

Cols = ['#e36414','#0db39e','#748cab','#8a5a44','#264653','#e9c46a']
#Cols2 = ['#e76f51', '#f4a261', '#e9c46a', '#2a9d8f', '#264653']
markers = ['o','s','d', 'v', '^', '>']
lines = ['-',':','--','-.', '','-']
def stylefun(G):
    return (Cols[G], markers[G], lines[G])

fig, ax= plt.subplots(figsize=(Col2,Col1*AR*1.2))     



#ax.set_ylim((ymin, ymax))
#ax.set_xlim((xmin, xmax))


kkcms = np.array([2**i for i in range(-1,8,2)])*1e-8
kkmuh = kkcms*3.6e11

batchlist = []
Cc = dfCapCrit.loc[dfCapCrit['Batch']=='211202_NMC', 'C-rate(prog)']
index2 = (Cc > 0.1) & (Cc < 4.8)






ms = 4
lw = 1.5
for i, (name, group) in enumerate(dfCapCrit.groupby(by=['Batch'])):
    
    df = group
    print(name)
    
    index = ~df['Thickness_lo(um)'].isnull() & ~df['Thickness_hi(um)'].isnull()
    
    if name == '211202_NMC': 
        index = index & index2
        
    label = name
    
    
    cc = df.loc[index,'C-rate_mean(1/h)'].to_numpy(dtype=float)
    tt = df.loc[index,'Thickness_max(um)'].to_numpy(dtype=float)
    
    print(tt)
    
    qa = np.array([])
    for it in tt:
        temp = dfall.loc[(dfall['Batch']==name) & (dfall['Thickness(um)']==it) & (dfall['C-rate(prog)']==0.1), 'Avg_DCapacity(mAh/cm2)'].mean()
        #if len(temp)>1:
        #    print(name + ' ' + str(it) + ' ' + str(temp))
        qa = np.append(qa, temp)
    
    qv = qa/(tt*1e-4)
    
    rl2 = cc*tt**2/3.6e11
    
    print(rl2)
    print(qv)
    
    ax.plot(qv, rl2*1e8, 
            marker = stylefun(i)[1], 
            linestyle = stylefun(i)[2], 
            color = stylefun(i)[0],  
            label = label, 
            markersize=4, 
            linewidth = 1)

ax.set_ylabel('RL$^2_d$ [10$^{-8}$ cm$^2$/s]')
ax.set_xlabel('volumetric capacity, Q$_v$ [mAh/cm3]')

ax.set_xlim((220,450))
ax.set_ylim((0,16))

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, 
          ncol = 2, 
          framealpha = 0,
          loc = 'best',
          fontsize = 9,
          columnspacing=0.5, 
          handletextpad = 0.3, 
          labelspacing = 0.3)



fig.tight_layout()
plt.show()


