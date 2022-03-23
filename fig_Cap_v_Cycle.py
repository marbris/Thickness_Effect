#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 15:48:21 2022

@author: martin
"""

#%%
import Cycler

import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
import main

SMALL_SIZE = 10
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
                            



dfall, dfCrit = main.dfall_dfCrit()


                    


#%% AreaCap vs Cycle



ProgSetting = 2


if ProgSetting == 0:
    Prog = 'Martin_cycles_1'
    NCol = 1
elif ProgSetting == 1:
    Prog = 'Martin_cycles_2'
    NCol = 2
elif ProgSetting == 2:
    Prog = 'Martin_cycles_3'
    NCol = 2

Cycfont = 8

ArMsSetting = 0

if ArMsSetting == 0:
    yCol = 'Avg_DCapacity(mAh/cm2)'
    yLab = 'Discharge Capacity \n[mAh/cm$^2$]'
    ymax = 9
    clabp = 1
    claby = 7
    clabrot = -30
    Leg_kwargs = {'loc': 'upper center', 'bbox_to_anchor' : (0.65,1.02)}

    
elif ArMsSetting == 1:
    yCol = 'Avg_DCapacity(mAh/gAM)'
    yLab = 'Discharge Capacity \n[mAh/g$_{AM}$]'
    ymax = 270
    clabp = 20 
    claby = 240
    clabrot = -5
    NCol = 1
    Leg_kwargs = {'loc': 'lower left', 'bbox_to_anchor' : (-0.02,-0.02)}
    

index = (dfall['Batch'] == '211202_NMC') & (dfall['Cycler_Program'] == Prog)
dfplot = dfall.loc[index, ('SampleID', 'Cycle_ID', 'C-rate(rnd)', 'Avg_C-rate(1/h)', 'Thickness(um)', 'Avg_DCapacity(mAh/cm2)', 'Avg_DCapacity(mAh/gAM)', 'Wet_Thickness(um)')].copy()

fig, ax= plt.subplots(figsize=(Col1,Col1*AR*1.2))   

plt.ylim((0,ymax))
plt.xlim((0,33)) 

Cols2 = ['#e76f51', '#f4a261', '#e9c46a', '#2a9d8f', '#264653']
markers = ['o', 'v', 's', '^', 'd']

def colfun(wt):
    if wt==600:
        col = Cols2[0]
    elif wt==500:
        col = Cols2[1]
    elif wt==400:
        col = Cols2[2]
    elif wt==300:
        col = Cols2[3]
    elif wt==200:
        col = Cols2[4]
    
    return col


def markerfun(i):
    return markers[np.mod(i,len(markers))]



for i, (name, group) in enumerate(dfplot.groupby(by=['SampleID'])):
    

    Thickness = group.loc[:,'Thickness(um)'].unique()[0]
    WT = group.loc[:,'Wet_Thickness(um)'].unique()[0]

    
    group.plot(x = 'Cycle_ID', 
               y=yCol, 
               label = '{}'.format(Thickness), 
               ax=ax, 
               marker = markerfun(i), 
               color = colfun(WT), 
               markersize = 3)
    



handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, 
          ncol = NCol,
          framealpha = 0,
          columnspacing=0.5, 
          handletextpad = 0.3,
          fontsize = 8,
          labelspacing = 0.3,
          **Leg_kwargs)
    
plt.xlabel('Cycle')
plt.ylabel(yLab) 

crts = dfplot['Avg_C-rate(1/h)'].unique().round(1)
crts = np.append(crts[~np.isnan(crts)], 0.1)    

Cycle_ID_Crate = [1.5, 4, 7, 10, 13, 16, 19, 22, 25, 27.5, 30.5]

for i, (name, group) in enumerate(dfplot.groupby(by=['C-rate(rnd)'])):
    
    
    y = group[yCol].max() + clabp
    if (i==9) | (i==0): 
        ax.text(Cycle_ID_Crate[i], y-clabp/2, str(crts[i]), color = 'k', fontsize = Cycfont, horizontalalignment = 'center')
    else:
        ax.text(Cycle_ID_Crate[i], y, str(crts[i]), color = 'k', fontsize = Cycfont, horizontalalignment = 'center')
    
    if i==0: 
        ax.text(Cycle_ID_Crate[-1], y-clabp/2, str(crts[i]), color = 'k', fontsize = Cycfont, horizontalalignment = 'center')
    
ax.text(8, claby, 'C-rates', color = 'k', fontsize = Cycfont, horizontalalignment = 'center', rotation = clabrot)    
    
fig.tight_layout()
plt.show()
    

# %%
