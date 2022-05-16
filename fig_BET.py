

#%%
import Cycler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Surface area

#get surface area from BET
File='Data/BET_analyzer/BET.csv'
df_BET = pd.read_csv(File)
index = df_BET['Good_BTS[Bol]']
#S = df_BET.loc[index,'BET_Surface_Area[m2/g]'].to_numpy(dtype=float)
#WT = df_BET.loc[index,'B[um]'].to_numpy(dtype=float)

fig, ax= plt.subplots(figsize=(4,3))

index = df_BET['Wet_Thickness[um]'].isin([900,1000,1100])

Cols2 = ['#e76f51', '#f4a261', '#e9c46a', '#2a9d8f', '#264653']

def colfun():
    return '#264653' #[ '#264653', '#264653', '#e76f51']

df_BET.loc[index].plot.bar(x='Wet_Thickness[um]', y = 'BET_Surface_Area[m2/g]', ax=ax, color = colfun())


ax.get_legend().remove()

ax.set_xlabel('Wet Thickness [$\mu$m]')
ax.set_ylabel('BET Surface Area [m$^2$/g]')
xticklabels = ax.get_xticklabels()
ax.set_xticklabels(xticklabels, rotation = 0)

ax.set_ylim((0, 16))

fig.tight_layout()
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Porosity

#get surface area from BET
File='Data/BET_analyzer/BET.csv'
df_BET = pd.read_csv(File)

fig, ax= plt.subplots(figsize=(4,3))

index = df_BET['Wet_Thickness[um]'].isin([900,1000,1100])

Cols2 = ['#e76f51', '#f4a261', '#e9c46a', '#2a9d8f', '#264653']

def colfun():
    return '#e76f51' #[ '#264653', '#264653', '#e76f51']

df_BET.loc[index].plot.bar(x='Wet_Thickness[um]', y = 'Porosity', ax=ax, color = colfun())


ax.get_legend().remove()

ax.set_xlabel('Wet Thickness [$\mu$m]')
ax.set_ylabel('Porosity')
xticklabels = ax.get_xticklabels()
ax.set_xticklabels(xticklabels, rotation = 0)

#ax.set_ylim((0, 16))

fig.tight_layout()
plt.show()