#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 09:48:48 2022

@author: martin
"""


import dfall_dfCrit
import pandas as pd
import numpy as np
import Cycler
import matplotlib.pyplot as plt


SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 15

Axes = {'titlesize': SMALL_SIZE,    # fontsize of the axes title
        'labelsize': MEDIUM_SIZE}   # fontsize of the x and y labels
plt.rc('axes', **Axes)


plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

#%%

dfall, dfCrit = dfall_dfCrit.dfall_dfCrit()




#%%

b=1
eta0=1
L=2

xl = lambda x: x/L
etafun = lambda x, L: 4*b*np.arctanh( np.exp(-x/L) * np.tanh( eta0/(4*b) ) )
Ifrac = lambda x, L: np.sinh( etafun(x,L)/b ) / np.sinh( eta0/b )



fig, ax= plt.subplots(figsize=(13,5))  


x=np.linspace(0,10,100)

critI=0.2

lcrit=np.array([])
ll = np.linspace(0,5,20)

for l in ll:
    y=Ifrac(x,l)
    plt.plot(x,y, label='L = {}'.format(l))
    
    lc = x[abs(y-critI)==min(abs(y-critI))]
    lcrit = np.append(lcrit, lc)
    
    print(lc)
    
    
#%%
    
    

fig, ax= plt.subplots(figsize=(13,5))  


plt.plot(ll[:-1],lcrit)

ax.set_ylabel('', fontsize = 16)
ax.set_xlabel('penetration depth', fontsize = 16)






