#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 18:05:52 2022

@author: martin
"""

#%%

import Cycler
from matplotlib import pyplot as plt
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
                       

#%% Charge discharge curves for all batteries. 

#fig, ax= plt.subplots(figsize=(Col1,Col1*AR*1.2))  
fig= plt.figure(figsize=(Col1,Col1*AR*1.2))
ax= fig.add_axes([0.15,0.2,0.8,0.75])

plt.ylim((2.2,4.45))
plt.xlim((-0.01,8.5))
#plt.xlim((-5,250))

ax.set_yticks([2.8, 3.6, 4.4])  

DataDirectory = "Data/"
BatchLabel = '211202_NMC'
PlotList = ['02', '13', '19']

SL = Cycler.get_SampleList(DataDirectory)
BI = Cycler.get_BatchInfo(BatchLabel)

tt=np.array([])
cc=np.array([])

plt_kwargs={'linewidth':1,
            'ax':ax}

Cols = ['#22223b', '#9a8c98', '#a4133c', '#ff4d6d']
linestyles = ['--', '-', '--', '-']

for SampleID in PlotList:
    
    tt = np.append(tt,BI['Samples'][SampleID]['ECCthickness']-BI['CurrentCollector']['Thickness'])
    
    
    df = Cycler.get_ChargeDischarge(BatchLabel, SampleID, DataDirectory)
    
    
    lw= 1
    Xs = 'Charge_Capacity(mAh/cm2)'
    #Xs = 'Charge_Capacity(mAh/gAM)'
    index = (df.loc[:,'Cycle_Index']==2) & (df.loc[:,'Step_Index']==2)
    df.loc[index,:].plot(y= 'Voltage(V)' , x=Xs,  
                         color = Cols[0], 
                         linestyle = linestyles[0],
                         **plt_kwargs)
    
    index = (df.loc[:,'Cycle_Index']==31) & (df.loc[:,'Step_Index']==52)
    df.loc[index,:].plot(y= 'Voltage(V)' , x=Xs,  
                         color = Cols[1],
                         linestyle = linestyles[1],
                         **plt_kwargs)
   
   
    Xs = 'Discharge_Capacity(mAh/cm2)'
    #Xs = 'Discharge_Capacity(mAh/gAM)'
    index = (df.loc[:,'Cycle_Index']==2) & (df.loc[:,'Step_Index']==4)
    df.loc[index,:].plot(y= 'Voltage(V)' , x=Xs,  
                         color = Cols[2],
                         linestyle = linestyles[2],
                         **plt_kwargs)
    
    index = (df.loc[:,'Cycle_Index']==31) & (df.loc[:,'Step_Index']==54)
    df.loc[index,:].plot(y= 'Voltage(V)' , x=Xs,  
                         color = Cols[3],
                         linestyle = linestyles[3],
                         **plt_kwargs)
    
    index = (df.loc[:,'Cycle_Index']==2) & (df.loc[:,'Step_Index']==4)
    xpos = df.loc[index,Xs].max()
    cc = np.append(cc,xpos)
   
 
    
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[-8:-4], 
          ['Charge #2', 'Charge #31', 'Discharge #2', 'Discharge #31'],
          loc='lower center', 
          framealpha = 0,
          columnspacing=0.5, 
          handletextpad = 0.3,
          labelspacing = 0.3,
          bbox_to_anchor = (0.5, -0.04),
          ncol = 2)


tt2, cc2 = zip(*sorted(zip(tt, cc), key=lambda k: k[1], reverse=True))

for i, t in enumerate(tt2):
    #ypos = 2.78 - np.mod(i,3)*0.09
    ypos = 2.75
    xpos = cc2[i]
    fs = SMALL_SIZE
    ax.text(xpos, ypos, "{:.0f} $\mu$m".format(t), color = 'k', fontsize = fs, horizontalalignment = 'center', verticalalignment = 'top')

#xpos = 0
xpos = 150
ax.text(xpos, 2.75, 'Cathode \nThickness [$\mu$m] :', color = 'k', fontsize = SMALL_SIZE, horizontalalignment = 'left', verticalalignment = 'top')

#xpos=1
xpos=100
ax.text(xpos, 3.1, 'C/10', color = 'k', fontsize = MEDIUM_SIZE, horizontalalignment = 'center')

ax.set_xlabel('Areal Capacity [mAh/cm$^2$]')
ax.set_ylabel('Voltage [V]')
    
#fig.tight_layout()
plt.show()
    
    