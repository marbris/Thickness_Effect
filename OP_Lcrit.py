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
import Cycler
import simplejson as json
import pandas as pd


#%%

#dfall, dfCapCrit, dfOPCrit = main.dfall_dfCapCrit(read_json = False, write_json = True)
dfall, dfCapCrit, dfOPCrit = main.dfall_dfCrit()


#%%

fig, ax= plt.subplots(figsize=(13,6))

ax.set_xlabel('Cycle')
ax.set_ylabel('Overpotential [V]')
#dfall['Batch']=='211202_NMC'
for name, group in dfall.groupby(by = ['Sample']):
    group.plot(x='Cycle', y='Overpotential(V)', marker='.', linestyle='-', ax=ax, label=name)
    

#%% Calculating diffusion depth

#dfplot = dfall.loc[dfall['Batch']=='211202_NMC'].copy()
#dfplot = dfall.loc[dfall['Batch']=='220203_NMC'].copy()
#dfplot = dfall.loc[dfall['Batch'].isin(['211202_NMC','220203_NMC'])].copy()

BatchList = ['211202_NMC','220203_NMC']

fig, ax= plt.subplots(figsize=(13,6))

Cols2 = ['#e76f51', '#f4a261', '#e9c46a', '#2a9d8f', '#264653']
markers = ['o', 'v', 's', '^']

def colfun(i):
    return Cols2[np.mod(i,len(Cols2))]


L0 = 57
ttss = np.array([])
wtt = np.array([])
ttt = np.array([])
for i, (name, group) in enumerate(dfall.loc[dfall['Batch'].isin(BatchList), :].groupby(by = ['Batch'])):

    tt = group['Thickness(um)'].unique()[0]
    wt = group['Wet_Thickness(um)'].unique()[0]
    
    ttss = np.append(ttss, tt+int(name[1])*1e-2)
    wtt = np.append(wtt, wt)
    ttt = np.append(ttt,tt)
    
    index = group['Cycle']<30
    ld = group.loc[index,'Avg_Penetration_Depth(x/L0)'].to_numpy(dtype=float)
    Ldiff = ld*L0
    cc = group.loc[index,'Avg_C-rate(1/h)'].to_numpy(dtype=float)
    
    ax.plot(cc, Ldiff, marker='.',linestyle='-', color=colfun(i), label=tt)
    #ax.plot(cc, Ldiff, marker='o',linestyle='-', color='g', label=tt)





#df = dfCapCrit.loc[dfCapCrit['Batch'].isin(['211202_NMC', '220203_NMC']), :].copy()

for name, group in dfCapCrit.loc[dfCapCrit['Batch'].isin(BatchList), :].groupby(by = ['Batch']):
    
    df = group

    Cc = df.loc[:, 'C-rate(prog)']
    #index2 = (Cc > 0.1) & (Cc < 4.8) 
    index = (Cc > 0.1) & (Cc < 4.8) & ~df['Thickness_lo(um)'].isnull() & ~df['Thickness_hi(um)'].isnull()

    xx = np.array(df.loc[index,'C-rate_mean(1/h)'].tolist())
    yy = np.array(df.loc[index,'Thickness_max(um)'].tolist())
        
    lo = np.array(df.loc[index,'Thickness_lo(um)'].tolist())
    hi = np.array(df.loc[index,'Thickness_hi(um)'].tolist())
    ee = np.vstack([lo,hi])

    ax.errorbar(xx, yy, yerr = ee, 
                    marker = 'o', 
                    linestyle = '-', 
                    color = 'r', 
                    capsize = 3, 
                    label = name, 
                    markersize=8, 
                    linewidth = 3, 
                    zorder = 110)


for name, df in dfOPCrit.loc[dfOPCrit['Batch'].isin(BatchList), :].groupby(by = ['Batch']):
    xx = df['Crit_C-rate(1/h)'].to_numpy(dtype=float)
    yy = df['Thickness(um)'].to_numpy(dtype=float)
        
    lo = df['Crit_C-rate_lo(1/h)'].to_numpy(dtype=float)
    hi = df['Crit_C-rate_hi(1/h)'].to_numpy(dtype=float)
    ee = np.vstack([lo,hi])

    ax.errorbar(xx, yy, xerr = ee, 
                    marker = 'o', 
                    linestyle = '-', 
                    color = 'k', 
                    capsize = 3, 
                    label = name, 
                    markersize=8, 
                    linewidth = 3, 
                    zorder = 110)



Leg_kwargs = {'loc': 'upper right', 'bbox_to_anchor' : (1.01, 1.05)}
Leg_Dummy_tt = -1
Leg_Col_Order = -1
Leg_Row_Order = -1

"""
NCol = len(np.unique(wtt))
maxrows = 4
for wti in np.unique(wtt):
    N_dummies = maxrows - len(wtt[wtt==wti])
    for i in range(N_dummies):
            ax.plot(np.zeros(1), np.zeros([1,2]), color='w', alpha=0, label=' ')
            wtt = np.append(wtt, wti)
            ttss = np.append(ttss, Leg_Dummy_tt)


handles, labels = ax.get_legend_handles_labels()
labels, handles, ttss2, wtt2 = zip(*sorted(zip(labels, handles, ttss, wtt), key=lambda k: (Leg_Col_Order*k[3], Leg_Row_Order*k[2]), reverse=False))
ax.legend(handles, labels, 
          ncol = NCol, 
          framealpha = 0, 
          columnspacing=0.7, 
          handletextpad = 0.3, 
          labelspacing = 0.3,
          **Leg_kwargs)
"""


ax.set_xlabel('C-rate(1/h)', fontsize=15)
ax.set_ylabel('penetration depth, L$_d$ [$\mu$m]', fontsize = 15)
handles, labels = ax.get_legend_handles_labels()

ax.set_xscale('log')

ax.legend(handles, labels, 
          loc = 'upper right', 
          ncol = 5)


#%%
n = 1 # 1
R = 8.314 # J/molK
T = 278 #K
F = 96485.3329 # C/mol
b=2*R*T/(n*F) #V

eta = lambda eta0, z: 4*b*np.arctanh(np.exp(-z)*np.tanh(eta0/(4*b)))

Jf = lambda eta0, z: np.sinh(eta(eta0,z)/b)/np.sinh(eta0/b)

thresh = 0.05#np.exp(-1)


fig, ax= plt.subplots(figsize=(13,6))

for name, group in dfall[dfall['Batch']=='211202_NMC'].groupby(by = ['Sample']):
    cc = group['Cycle'].to_numpy()
    nn = group['Overpotential(V)'].to_numpy()
    zcrt = np.array([])
    for nni in nn:
        if ~np.isnan(nni):
            fun = lambda z: abs(Jf(nni, z) - thresh)
            Zcrit = fmin(fun,np.array([1]))
            zcrt = np.append(zcrt,Zcrit)
        else:
            zcrt = np.append(zcrt,np.nan)
    
    ax.plot(cc, zcrt, marker='.', linestyle='-', label = name)    
    
    #group.plot(x='Cycle_ID', y='Overpotential(V)', marker='.', linestyle='-', ax=ax, label=name)

ax.set_xlabel('Cycle')
ax.set_ylabel('Critical Thickness, x/L')
handles, labels = ax.get_legend_handles_labels()

ax.legend(handles, labels, loc = 'lower right')
    
#%%

fig, ax= plt.subplots(figsize=(13,6))
zz = np.linspace(0,2,1000)
fun = lambda z: abs(Jf(0.0247, z) - 0)
plt.plot(zz,fun(zz))


# %%



dfplot = dfall.loc[dfall['Batch']=='M. Singh (2016), NMC'].copy()

fig, ax= plt.subplots(figsize=(10,6))


for name, group in dfplot.groupby(by = ['Cycle']):
    label = group['C-rate(1/h)'].unique()[0]
    group.plot(x='Thickness(um)',y='Capacity(mAh/cm2)', marker = 'o', label = label, ax = ax)


ax.set_ylabel('Capacity(mAh/cm2)')
ax.set_xlabel('Thickness(um)')




#%%

