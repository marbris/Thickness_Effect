#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 16:01:11 2022

@author: martin
"""
#%%

import dfall_dfCrit
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.optimize import fmin

#%%

#dfall, dfCrit = dfall_dfCrit.dfall_dfCrit(read_json = False, write_json = False)
dfall, dfCrit = dfall_dfCrit.dfall_dfCrit()

#%%

fig, ax= plt.subplots(figsize=(13,6))

ax.set_xlabel('Cycle_ID')
ax.set_ylabel('Overpotential [V]')

for name, group in dfall[dfall['Batch']=='211202_NMC'].groupby(by = ['SampleID']):
    group.plot(x='Cycle_ID', y='Overpotential(V)', marker='.', linestyle='-', ax=ax, label=name)
    

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

for name, group in dfall[dfall['Batch']=='211202_NMC'].groupby(by = ['SampleID']):
    cc = group['Cycle_ID'].to_numpy()
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

ax.set_xlabel('Cycle_ID')
ax.set_ylabel('Critical Thickness, x/L')
handles, labels = ax.get_legend_handles_labels()

ax.legend(handles, labels, loc = 'lower right')
    
#%%

fig, ax= plt.subplots(figsize=(13,6))
zz = np.linspace(0,2,1000)
fun = lambda z: abs(Jf(0.0247, z) - 0)
plt.plot(zz,fun(zz))


# %%
