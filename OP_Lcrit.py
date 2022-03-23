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

#%%

#dfall, dfCrit = main.dfall_dfCrit(read_json = False, write_json = True)
dfall, dfCrit = main.dfall_dfCrit()

#%%

fig, ax= plt.subplots(figsize=(13,6))

ax.set_xlabel('Cycle_ID')
ax.set_ylabel('Overpotential [V]')

for name, group in dfall[dfall['Batch']=='211202_NMC'].groupby(by = ['SampleID']):
    group.plot(x='Cycle_ID', y='Overpotential(V)', marker='.', linestyle='-', ax=ax, label=name)
    

#%% Calculating diffusion depth

dfplot = dfall.loc[dfall['Batch']=='211202_NMC'].copy()

fig, ax= plt.subplots(figsize=(13,6))

n = 1 # 1
R = 8.314 # J/molK
T = 278 #K
F = 96485.3329 # C/mol
b=2*R*T/(n*F) #V



xmin = 1e-1
xmax = 2.1e1

#ymin = 0
#ymax = 200

#ax.set_ylim((ymin, ymax))
ax.set_xlim((xmin, xmax))


eta_d = lambda Jd, eta0: b*np.arcsinh(Jd*np.sinh(eta0/b)) 
Ld = lambda L0, Jd, eta0: np.log( np.tanh( eta0/(4*b) )/np.tanh( eta_d(Jd,eta0)/(4*b) ))

Cols2 = ['#e76f51', '#f4a261', '#e9c46a', '#2a9d8f', '#264653']
markers = ['o', 'v', 's', '^']

def colfun(wt):
    if wt==600:
        col = Cols2[0]
    elif wt==500:
        col = Cols2[1]
    elif wt==400:
        col = Cols2[2]
    elif wt==300:
        col = Cols2[3]
    elif wt==200:
        col = Cols2[4]
    else:
        col ='k'
    
    return col

def markerfun(wt, ttss):
    
    
    dftemp = dfplot.loc[dfplot['Wet_Thickness(um)']==wt, ('Thickness(um)', 'SampleID')].drop_duplicates()
    
    ss_arr = np.array(list(map(int, dftemp.loc[:,'SampleID'].tolist())))
    tt_arr = np.array(dftemp.loc[:,'Thickness(um)'].tolist())
    
    ttss_arr = tt_arr + ss_arr*1e-2
    #the number of thicker samples within this wet thickness
    N_Thicker = sum(ttss_arr>ttss)
    #print((ttss_arr, ttss, N_Thicker))

    #the marker is determined by how many thicker samples there are
    mark = markers[N_Thicker]
    
    return mark

Jd = np.exp(-1)
L0 = 93

ttss=np.array([])
wtt=np.array([])
ccc = np.array([])
ttt = np.array([])
for name, group in dfplot.groupby(by = ['SampleID']):
    cc = group['Avg_C-rate(1/h)'].to_numpy(dtype=float)
    cce = group['Std_C-rate(1/h)'].to_numpy(dtype=float)
    nn = group['Avg_Overpotential(V)'].to_numpy(dtype=float)
    nne = group['Std_Overpotential(V)'].to_numpy(dtype=float)
    
    #im ignoring all points after the first nan, including. 
    nani = np.argmax(np.isnan(nn))-1
    cc = np.unique(cc[:nani])
    cce = np.unique(cce[:nani])
    nn = np.unique(nn[:nani])
    nne = np.unique(nne[:nani])
    
    ld = Ld(L0,Jd,nn)
    ldep = abs(ld - Ld(L0,Jd,nn+nne))
    lden = abs(ld - Ld(L0,Jd,nn-nne))
    lde = np.stack([lden,ldep])
    
    tt = group['Thickness(um)'].unique()[0]
    wt = group['Wet_Thickness(um)'].unique()[0]
    
    ttss = np.append(ttss, tt+int(name)*1e-2)
    wtt = np.append(wtt, wt)
    ttt = np.append(ttt,tt)
    
    Ldiff = ld*L0
    
    Cc = np.interp(tt,np.flip(Ldiff),np.flip(cc))
    ccc = np.append(ccc,Cc)
    
    ax.plot(Cc, tt, marker=markerfun(wt,ttss[-1]), markerfacecolor=colfun(wt), markersize=15, alpha=0.5, color = 'k', zorder =100)
    ax.hlines(tt,0.1,Cc,color = colfun(wt), linestyle = '--', zorder = 0)    
    
    ax.errorbar(cc, Ldiff, yerr=lde, marker=markerfun(wt,ttss[-1]),linestyle='-', color=colfun(wt), capsize=5, label=tt)




df = dfCrit.loc[dfCrit['Batch']=='211202_NMC', :].copy()


Cc = df.loc[:, 'C-rate(rnd)']


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
                color = '#e36414', 
                capsize = 3, 
                label = 'Critical Thickness', 
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
