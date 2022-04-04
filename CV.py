
#%%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


#%%

filepath = 'Data/CV/220209_08_CV.DTA'; skiprows = 58
#filepath = 'Data/CV/220209_08_CV_2.DTA'; skiprows = 57

df = pd.read_csv(filepath, 
                 delimiter = '\t', 
                 skiprows = skiprows, 
                 header = None, 
                 usecols = [1,2,3,4,5,6,7,8,9,10,11],
                 names = ['Pt', 'T(s)', 'Vf(V vs Ref)', 'I(A)',	'Vu(V)', 'Sig(V)', 'Ach(V)', 'IERange', 'Over(bits)', 'Cycle', 'Temp(C)'],
                 index_col = 'Pt')

df.loc[:,'I(A)'] = df['I(A)']*1e3
df = df.rename(columns={'I(A)': 'I(mA)'})


#%%

fig, ax= plt.subplots(figsize=(13,6))

#df.plot(x='T(s)', y = 'Vf(V vs Ref)', ax=ax)
#df.plot(y='Vf(V vs Ref)', x = 'I(mA)', ax=ax)


#df[df['Cycle']==1].plot(x='Vf(V vs Ref)', y = 'I(mA)', ax=ax, marker = '.')



#ax.set_xscale('log')

prominence = 0.005


index_cycle = df['Cycle']==1
MaxV = df.loc[index_cycle, 'Vf(V vs Ref)'].max()
index_maxV =df.loc[(index_cycle) & (df['Vf(V vs Ref)']==MaxV)].index[0]



Index_ch = index_cycle & (df.index <= index_maxV)
I_ch = df.loc[Index_ch, 'I(mA)'].to_numpy(dtype=float)
V_ch = df.loc[Index_ch, 'Vf(V vs Ref)'].to_numpy(dtype=float)
pks_ch, props_ch = find_peaks(I_ch, prominence = prominence)

Index_dch = index_cycle & (df.index >= index_maxV)
I_dch = df.loc[Index_dch, 'I(mA)'].to_numpy(dtype=float)
V_dch = df.loc[Index_dch, 'Vf(V vs Ref)'].to_numpy(dtype=float)
pks_dch, props_dch = find_peaks(-I_dch, prominence = prominence)

ax.plot(V_ch, I_ch, marker='.', color='b')
ax.plot(V_dch, I_dch, marker='.', color='g')



pks = np.append(pks_ch, pks_dch + len(V_ch))

I = df.loc[df['Cycle']==1, 'I(mA)'].to_numpy(dtype=float)
Y = I[pks]
X = df.loc[df['Cycle']==1, 'Vf(V vs Ref)'].to_numpy(dtype=float)
X = X[pks]
ax.plot(X, Y, marker='o', color='r')

#%%

Index_ch = index_cycle & (df.index <= index_maxV)
I_ch = df.loc[Index_ch, 'I(mA)'].to_numpy(dtype=float)
V_ch = df.loc[Index_ch, 'Vf(V vs Ref)'].to_numpy(dtype=float)
pks_ch, props_ch = find_peaks(I_ch, prominence = prominence)

V_ch_pk = V_ch[pks_ch[-1]]


Index_dch = index_cycle & (df.index >= index_maxV)
I_dch = df.loc[Index_dch, 'I(mA)'].to_numpy(dtype=float)
V_dch = df.loc[Index_dch, 'Vf(V vs Ref)'].to_numpy(dtype=float)
pks_dch, props_dch = find_peaks(-I_dch, prominence = prominence)

V_dch_pk = V_dch[pks_dch[0]]



I = I_dch[:pks_dch[0]]
eta = V_dch[:pks_dch[0]] - (V_ch_pk + V_dch_pk)/2

fig, ax= plt.subplots(figsize=(13,6))


ax.plot(abs(I), eta, marker='.')

ax.set_xscale('log')
