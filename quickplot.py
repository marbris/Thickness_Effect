#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 16:37:41 2022

@author: martin
"""

import dfall_dfCrit
import pandas as pd
import numpy as np
import Cycler
import matplotlib.pyplot as plt


plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels

#%%

dfall, dfCrit = dfall_dfCrit.dfall_dfCrit()



#%%

fig, ax= plt.subplots(figsize=(13,5))    

df = dfall.loc[dfall['Batch']=='220203_NMC'].copy()


df.loc[df['Cycle_ID']==1].plot(x='Thickness(um)', y='Capacity(mAh/cm2)', ax=ax, marker = 'o', linestyle = '', label = 'Today, 0.1C')


df = dfall.loc[dfall['Batch']=='211202_NMC'].copy()


crts2 = [0.1, 0.6, 1.0, 1.3,1.6, 2.1, 2.4, 3.6, 5.6, 6.4]

cr_groups = np.array([1.0, 3.6, 13.1])
Cols = ['#219ebc', '#e36414', '#606c38']
markers = ['^','o', 'v']

def colorfun(G):
    #the color is determined by how many lower cr-groups there are
    color = Cols[G]
    return color
    
def sizefun(G, cr):
    if G == 0:
        ms = 1 + 4*cr
    elif G==1:
        ms = 2*cr
    elif G==2:
        ms = (cr-4.5)*3    
    return ms

def markfun(G):
    return markers[G]



for name, group in df.groupby(by=['C-rate(rnd)']):
    if name in crts2:
        cr=name
        G = int(3 - sum(cr <= cr_groups))
        group.plot(x='Thickness(um)', 
                   y='Avg_DCapacity(mAh/cm2)', 
                   marker = markfun(G), 
                   markersize = sizefun(G, cr), 
                   color = colorfun(G),
                   label = name , 
                   ax=ax)


handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, 
          ncol = 1, 
          framealpha = 0,  
          loc = 'lower right',
          columnspacing=0.5, 
          handletextpad = 0.3, 
          labelspacing = 0.3,
          fontsize = 15)


ax.set_ylabel('Areal Discharge Capacity \n[mAh/cm$^2$]', fontsize = 16)
ax.set_xlabel('Thickness [$\mu m$]', fontsize = 16)

fig.tight_layout()
plt.show()

 