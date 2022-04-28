
#%%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import Cycler


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

plt.rc('figure', dpi = 100)

Col1= 8.9/2.54 # 8.9cm single column figure width in Nature
Col2= 18.3/2.54 # 18.3cm double column figure width in Nature

AR = 1/1.618


filepath = 'Data/CV/220209_08_CV_MITS.json'

df = Cycler.get_PointData('','',filepath = filepath)  




#%%

fig, ax= plt.subplots(figsize=(13,6))

#ax2 = ax.twinx()
#df.plot(x='Test_Time(s)', y = 'Voltage(V)', ax=ax, color ='k')
#df.plot(x='Test_Time(s)', y = 'Current(mA)', ax=ax2, color = 'r')

df.plot(x='Voltage(V)', y = 'Current(mA)', ax=ax, color = 'r')

#df[df['Cycle']==1].plot(x='Voltage(V)', y = 'Current(mA)', ax=ax, marker = '.')
plt.show()

#%%

fig, ax= plt.subplots(figsize=(13,6))

#ax.set_xscale('log')

prominence = 0.005

Cyc_i = 3
Rpot = np.array([])
for Cyc_i in range(2,12):

    index_cycle = df['Cycle']==Cyc_i
    MinV = df.loc[index_cycle, 'Voltage(V)'].min()
    index_minV =df.loc[(index_cycle) & (df['Voltage(V)']==MinV)].index[0]



    Index_ch = index_cycle & (df.index >= index_minV)
    I_ch = df.loc[Index_ch, 'Current(mA)'].to_numpy(dtype=float)
    V_ch = df.loc[Index_ch, 'Voltage(V)'].to_numpy(dtype=float)
    pks_ch, props_ch = find_peaks(I_ch, prominence = prominence)

    Index_dch = index_cycle & (df.index <= index_minV)
    I_dch = df.loc[Index_dch, 'Current(mA)'].to_numpy(dtype=float)
    V_dch = df.loc[Index_dch, 'Voltage(V)'].to_numpy(dtype=float)
    pks_dch, props_dch = find_peaks(-I_dch, prominence = prominence)

    ax.plot(V_ch, I_ch, marker='.', color='b')
    ax.plot(V_dch, I_dch, marker='.', color='g')



    pks = np.append(pks_dch, pks_ch + len(V_dch))

    I = df.loc[df['Cycle']==Cyc_i, 'Current(mA)'].to_numpy(dtype=float)
    Y = I[pks]
    X = df.loc[df['Cycle']==Cyc_i, 'Voltage(V)'].to_numpy(dtype=float)
    X = X[pks]
    ax.plot(X, Y, marker='o', color='r')
    
    Rpot = np.append(Rpot,np.mean([X[0],X[-1]]))
    
    plt.vlines(np.mean([X[0],X[-1]]),-0.15,0.15, color='k',)
    


ax.set_xlabel('Voltage (V)', fontsize=18)
ax.set_ylabel('Current(mA)', fontsize=18)
ax.tick_params(axis='x', labelsize=15)

ax.tick_params(axis='y', labelsize=15)

fig.tight_layout()
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig, ax= plt.subplots(figsize=(13,6))

plt.plot(Rpot, marker = 'o', color ='r')

print('Equilibrium Voltage : {} V '.format(np.mean(Rpot[:-1])))
print('Equilibrium Voltage std : {} V '.format(np.std(Rpot[:-1])))

plt.show()
#%%

Index_ch = index_cycle & (df.index <= index_maxV)
I_ch = df.loc[Index_ch, 'Current(mA)'].to_numpy(dtype=float)
V_ch = df.loc[Index_ch, 'Voltage(V)'].to_numpy(dtype=float)
pks_ch, props_ch = find_peaks(I_ch, prominence = prominence)

V_ch_pk = V_ch[pks_ch[-1]]


Index_dch = index_cycle & (df.index >= index_maxV)
I_dch = df.loc[Index_dch, 'Current(mA)'].to_numpy(dtype=float)
V_dch = df.loc[Index_dch, 'Voltage(V)'].to_numpy(dtype=float)
pks_dch, props_dch = find_peaks(-I_dch, prominence = prominence)

V_dch_pk = V_dch[pks_dch[0]]



I = I_dch[:pks_dch[0]]
eta = V_dch[:pks_dch[0]] - (V_ch_pk + V_dch_pk)/2

fig, ax= plt.subplots(figsize=(13,6))


ax.plot(abs(I), eta, marker='.')

ax.set_xscale('log')
