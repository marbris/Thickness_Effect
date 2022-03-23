#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 16:01:11 2022

@author: martin
"""
#%%

import main
import matplotlib.pyplot as plt
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


#dfall, dfCrit = main.dfall_dfCrit(read_json = False, write_json = True)
dfall, dfCrit = main.dfall_dfCrit()


Batch = '211202_NMC'
Sample = '16'

#xx = np.array([])
cycles = np.array([3,6,9])


#%% Calculating diffusion depth

dfplot = dfall.loc[dfall['Batch']=='211202_NMC'].copy()

fig= plt.figure(figsize=(Col2,Col1*AR*1.2))
ax= fig.add_axes([0.15,0.2,0.8,0.75])


n = 1 # 1
R = 8.314 # J/molK
T = 278 #K
F = 96485.3329 # C/mol
b=2*R*T/(n*F) #V




Jd = 0.05
L0 = 93




etafun = lambda x, eta0: 4*b*np.arctanh( np.exp(-x/L0) * np.tanh(eta0/(4*b)) )
Ifrac = lambda x, eta0: np.sinh(etafun(x,eta0)/b)/np.sinh(eta0/b)



markers = ['o', 'v', 's', '^']

Cols = ['#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6', '#ee9b00', '#ca6702', '#bb3e03', '#ae2012', '#9b2226']

def colfun(i):
    return Cols[np.mod(i-1,len(Cols))]



X = np.linspace(0,2*L0,1000)

for Cyc in cycles:
    
    eta0 = dfplot.loc[dfplot['Cycle_ID']==Cyc, 'Overpotential(V)'].values[0]
    Y = Ifrac(X,eta0)
    ax.plot(X,Y,color=colfun(Cyc),linestyle='-')
    
    Ld = min(Y-Jd)
    
    plt.hlines(Jd,0,)
    



#ax.set_ylim((0, 1.1))
ax.set_xlim((0, max(X)))

ax.set_xlabel('Distance from Separator, [$\mu$m]')
ax.set_ylabel('Fractional \nreaction rate current')

