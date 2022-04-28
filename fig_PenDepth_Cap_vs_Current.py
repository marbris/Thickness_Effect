#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 25 20:09 2022

@author: martin
"""
#%%

import main
import matplotlib.pyplot as plt
import numpy as np
import Cycler
import pandas as pd

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

SampleList = {
                            #'220203_NMC': ['10', '11', '15', '16', '24', '25'],
                            #'220203_NMC': ['03', '04', '05', '06', '09', '17', '18', '21', '22', '23'],
                            '220203_NMC': ['03', '06', '18', '21', '22', '23'],
                            '211202_NMC': ['02', '03', '05', '06', '07', '08', '09', '12', '13', '15', '16', '17', '18', '19']
                        }

#dfall, dfCapCrit, dfOPCrit = main.dfall_dfCapCrit(read_json = False, write_json = True)
dfall, dfCapCrit, dfOPCrit = main.dfall_dfCrit()

#%%



fig, ax= plt.subplots(figsize=(Col2,Col1*AR*1.2))   

ax2 = ax.twinx()

xmin= 1e-1
xmax = 4e1

ax.set_xlim(xmin, xmax)

Cols2 = ['#e76f51', '#f4a261', '#e9c46a', '#2a9d8f', '#264653']
markers = ['o', 'v', 's', '^']
def colfun(i):
    return Cols2[np.mod(i,len(Cols2))]
L0 = 57

Jd = 0.0001
n = 1 # 1
R = 8.314 # J/molK
T = 298 #K
F = 96485.3329 # C/mol
b=2*R*T/(n*F) #V
eta_d = lambda eta0: b*np.arcsinh(Jd*np.sinh(eta0/b)) 
Ld = lambda eta0: np.log( np.tanh( eta0/(4*b) )/np.tanh( eta_d(eta0)/(4*b) ))


index1 = (dfall['Batch']=='211202_NMC') & \
         (dfall['Avg_DCapacity(mAh/gAM)'] > 0.05) & \
         (dfall['Sample'] != '0') 

for i, (sample, df) in enumerate(dfall.loc[index1].groupby(by = 'Sample')):
    
    Cyc_index = df['Cycle']<30
    X = df.loc[Cyc_index,'Avg_Current(mA/cm2)'].to_numpy(dtype=float)
    
    #ld = df.loc[Cyc_index,'Avg_Penetration_Depth(x/L0)'].to_numpy(dtype=float)
    ld = df.loc[Cyc_index,'Avg_Overpotential(V)'].apply(Ld).to_numpy(dtype=float)
    tt = df['Thickness(um)'][0]
    Y1 = ld*L0
    ax.plot(X, Y1, marker='.',linestyle='-', color='k', label = '', markerfacecolor = 'w')
    
    #index = Cyc_index & df[]
    
    #Y2 = df.loc[Cyc_index,'Avg_DCapacity(mAh/cm2)'].to_numpy(dtype=float)
    Y2 = df.loc[Cyc_index,'Avg_DCapacity(mAh/gAM)'].to_numpy(dtype=float)
    ax2.plot(X, Y2, marker='.',linestyle='-', color='r', label = '')



ax.hlines(1,xmin,xmax,color = 'k', linestyle = ':')
ax.text(xmin*1.05, 0.9, 'Penetration = Thickness', fontsize = 10, verticalalignment = 'top')
    
ax.set_xlabel('Discharge Current Density [mA/cm$^2$]')    
ax.set_ylabel('Excess \nPenetration depth')    
ax2.set_ylabel('Areal Capacity \n[mAh/cm$^2$]', color = 'r') 

ax.set_xscale('log')

fig.tight_layout()
plt.show()
# %%
