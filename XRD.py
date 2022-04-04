#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 17:27:29 2022

@author: martin
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import simplejson as json
from scipy.signal import find_peaks


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
                

#%%


datafile = 'Data/XRD/NMC_3_GM/NMC_3_GM_4tt35tt_05ts_6ks_4_good/NMC_3_GM_4tt35tt_05ts_6ks_4_bg_subtracted.xye'
df_NMC = pd.read_csv(datafile, sep = ' ', skiprows = 1, header = None, names = ['2Th_[Mo]', 'Cnt', 'e'])

patternfile = 'Data/XRD/COD9005350_NMC.csv'
df_Pat = pd.read_csv(patternfile, sep = ',', header = 0, names = ['d(A)', '2Th_[Mo]', 'I', 'h', 'k', 'l', 'Comment'])

#%%
l_Cu = 1.54056 #Ang
l_Mo = 0.70930 #Ang

df_NMC.loc[:,'2Th_[Cu]'] = np.arcsin( l_Cu/l_Mo * np.sin( df_NMC['2Th_[Mo]']*(np.pi/180) / 2 ) ) * 2 * 180/np.pi
df_Pat.loc[:,'2Th_[Cu]'] = np.arcsin( l_Cu/l_Mo * np.sin( df_Pat['2Th_[Mo]']*(np.pi/180) / 2 ) ) * 2 * 180/np.pi

fig, ax= plt.subplots(figsize=(Col1,Col1*AR*1.2))

df_NMC.plot(x='2Th_[Cu]', y='Cnt', ax=ax, color ='k', marker = None, linewidth = 0.5, label = "NMC-powder")


Pat_2th = df_Pat['2Th_[Cu]'].to_numpy(dtype=float)
Pat_I = df_Pat['I'].to_numpy(dtype=float)
plt.vlines(Pat_2th, 0, Pat_I/max(Pat_I)*20, color = 'r', linewidth = 1, zorder=1e3, alpha = 0.7, label = 'pattern')


ax.set_ylabel('Counts')
ax.set_xlabel('2$\Theta$ [$^\circ$]')

#ax.set_ylim((-0.5, 9))
ax.set_xlim((10, 80))


handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ['XRD pattern', 'NMC peaks'], 
          loc = 'upper right',
          framealpha = 0,
          #fontsize = 8,
          bbox_to_anchor=(1.03,1.03), 
          handletextpad = 0.3, 
          labelspacing = 0.3)


fig.tight_layout()
plt.show()







#%%
xrdfile = 'Data/XRD/mp-25411_xrd_Cu.json'
with open(xrdfile) as file:
    xrd = json.load(file)
    
peaks_NMC = [878,2454,2571,2598,3121,3473,4300,4758,4793,5051]
tth = df_NMC.loc[peaks_NMC,'2Th_[Cu]'].values
dd_NMC = l_Cu/(2*np.sin(tth/2*np.pi/180))

ymin=0
aa = np.array([])
ff = np.array([])
pp = np.array([])
hh = np.array([])
dd_LCO = np.array([])
for peak in xrd['pattern']:
    d = peak[3]
    h=peak[1][0]
    k=peak[1][1]
    j=peak[1][2]
    l=peak[1][3]
    f = np.sqrt(4/3*(h**2+k**2+h*k) + l**2) #a=c
    
    aa = np.append(aa,d*f)
    ff = np.append(ff,f)
    dd_LCO = np.append(dd_LCO,d)
    hh = np.append(hh,peak[0])
    pp = np.append(pp,peak[2])
    
kk = dd_NMC/dd_LCO[:10]

    

pp2 = 2*np.arcsin(1/kk[1]*np.sin(pp/2*np.pi/180))*180/np.pi


plt.vlines(pp, 0, hh/5, color = 'r', linewidth = 1, zorder=1e3, alpha = 0.7, label = 'pattern')
#peaks, peakdict = find_peaks(df.loc[:,'Cnt'], prominence = (3, 1e5))

#df.loc[peaks,:].plot(x='2Th_[Cu]', y='Cnt', ax=ax, color ='k', marker = 'o', linestyle='')

ax.set_ylabel('Counts')
ax.set_xlabel('2$\Theta$ [$^\circ$]')

#ax.set_ylim((-0.5, 9))
ax.set_xlim((10, 80))


handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ['XRD pattern', 'NMC peaks'], 
          loc = 'upper right',
          framealpha = 0,
          #fontsize = 8,
          bbox_to_anchor=(1.03,1.03), 
          handletextpad = 0.3, 
          labelspacing = 0.3)


fig.tight_layout()
plt.show()




#%%



fig, ax= plt.subplots(figsize=(13,6))
plt.plot(df.loc[:,'Cnt'].values) 

#%%

