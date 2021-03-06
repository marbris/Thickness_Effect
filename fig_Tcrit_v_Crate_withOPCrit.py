#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 16:01:11 2022

@author: martin
"""
#%%

import main
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.optimize import fmin
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

#%%

#dfall, dfCapCrit, dfOPCrit = main.dfall_dfCrit(read_json = False, write_json = True)
dfall, dfCapCrit, dfOPCrit = main.dfall_dfCrit()

    

#%% Calculating diffusion depth

#dfplot = dfall.loc[dfall['Batch']=='211202_NMC'].copy()
#dfplot = dfall.loc[dfall['Batch']=='220203_NMC'].copy()



fig, ax= plt.subplots(figsize=(Col2,Col1*AR*1.2))   

n = 1 # 1
R = 8.314 # J/molK
T = 278 #K
F = 96485.3329 # C/mol
b=2*R*T/(n*F) #V

sig_l = 9.169e-3 # S/cm
i0 = 7.2e-4 # A/cm2
S = 9.3e3 # cm2/cm3
rho_l = 1/sig_l #Ohm cm

#L0 = np.sqrt(b/(2*i0*S*rho_l))*1e4 #um
L0 = 57

xmin = 1e-1
xmax = 3e1

ymin = 0
ymax = 350

ax.set_ylim((ymin, ymax))
ax.set_xlim((xmin, xmax))

Cols = ['#e36414','#0db39e','#748cab','#8a5a44','#264653','#e9c46a']
#Cols2 = ['#e76f51', '#f4a261', '#e9c46a', '#2a9d8f', '#264653']
markers = ['o','s','d', 'v', '^', '>']
lines = ['-',':','--','-.', '','-']
def stylefun(G):
    return (Cols[G], markers[G], lines[G])


#df = dfCrit.loc[dfCrit['Batch'].isin(['211202_NMC', '220203_NMC']), :].copy()

Cc = dfCapCrit.loc[dfCapCrit['Batch']=='211202_NMC', 'C-rate(prog)']
index2 = (Cc > 0.1) & (Cc < 4.8) 

#legsort = []
#LegOrd = 100

for i, (batch, df) in enumerate(dfCapCrit.groupby(by=['Batch'])):

    
    #legsort.append(i+LegOrd)

    Cc = df.loc[:, 'C-rate(prog)']


    #index2 = (Cc > 0.1) & (Cc < 4.8) 
    index = (Cc > 0.1) & (Cc < 4.8) & ~df['Thickness_lo(um)'].isnull() & ~df['Thickness_hi(um)'].isnull()
    
    index = ~df['Thickness_lo(um)'].isnull() & ~df['Thickness_hi(um)'].isnull()
    
    if batch == '211202_NMC': 
        index = index & index2

    xx = np.array(df.loc[index,'C-rate_mean(1/h)'].tolist())
    yy = np.array(df.loc[index,'Thickness_max(um)'].tolist())
        
    lo = np.array(df.loc[index,'Thickness_lo(um)'].tolist())
    hi = np.array(df.loc[index,'Thickness_hi(um)'].tolist())
    ee = np.vstack([lo,hi])

    ax.errorbar(xx, yy, yerr = ee, 
                    marker = stylefun(i)[1], 
                    linestyle = stylefun(i)[2], 
                    color = stylefun(i)[0], 
                    capsize = 2, 
                    label = batch, 
                    markersize=4, 
                    linewidth = 1.5, 
                    zorder = 110)


BatchList = ['211202_NMC', '220203_NMC']

#LegOrd = 0

for i, (batch, df) in enumerate(dfOPCrit[dfOPCrit['Batch'].isin(BatchList)].groupby(by=['Batch'])):
    
    
    #legsort.append(i+LegOrd)
    

    xx = np.array(df.loc[:,'Crit_C-rate(1/h)'].tolist())
    yy = np.array(df.loc[:,'Thickness(um)'].tolist())
    
    lo = np.array(df.loc[:,'Crit_C-rate_lo(1/h)'].tolist())
    hi = np.array(df.loc[:,'Crit_C-rate_hi(1/h)'].tolist())
    
    i_sort = np.argsort(xx)
    xx = xx[i_sort]
    yy = yy[i_sort]
    lo = lo[i_sort]
    hi = hi[i_sort]
    
    ee = np.vstack([lo,hi])

    ax.errorbar(xx, yy, xerr = ee, 
                    marker = stylefun(i)[1], 
                    linestyle = stylefun(i)[2], 
                    color = stylefun(i)[0],
                    markerfacecolor = 'w',
                    capsize = 2, 
                    label = batch, 
                    markersize=4.5, 
                    linewidth = 1, 
                    zorder = 120)



ax.set_xlabel('C-rate [1/h]')
ax.set_ylabel('Cathode Thickness [$\mu$m]')
#handles, labels = ax.get_legend_handles_labels()

ax.set_xscale('log')



#LegOrd = 50

for i in range(4):
    #legsort.append(i+LegOrd)
    ax.plot(np.zeros(1), np.zeros(1), color='w', alpha=0, label=' ')
    


handles, labels = ax.get_legend_handles_labels()

legsort = [2]*4 + [3]*2 + [4]*4 + [1]*2

handles2, labels2, legsort2 = zip(*sorted(zip(handles, labels, legsort), key=lambda k: k[2], reverse=False))

ax.legend(handles2, labels2, 
          ncol = 2, 
          framealpha = 0, 
          columnspacing=4, 
          handletextpad = 0.3, 
          labelspacing = 0.3,
          loc = 'upper right',
          bbox_to_anchor = (1.00, 0.87))

plt.text(7, 345, 'Optimal Thickness', 
         fontweight = 'bold',
         verticalalignment = 'top')
plt.text(2, 345, 'Penetration Depth \n($I = 0.01 I_0$)', 
         fontweight = 'bold',
         verticalalignment = 'top')

#ax.legend(handles, labels, 
#          loc = 'upper right', 
#          ncol = 2)

fig.tight_layout()
plt.show()

#%%
