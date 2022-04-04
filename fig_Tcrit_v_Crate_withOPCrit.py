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

#%%

#dfall, dfCrit = main.dfall_dfCrit(read_json = False, write_json = True)
dfall, dfCrit = main.dfall_dfCrit()

    

#%% Calculating diffusion depth

#dfplot = dfall.loc[dfall['Batch']=='211202_NMC'].copy()
#dfplot = dfall.loc[dfall['Batch']=='220203_NMC'].copy()
dfplot = dfall.loc[dfall['Batch'].isin(['211202_NMC','220203_NMC'])].copy()

fig, ax= plt.subplots(figsize=(Col2,Col1*AR*1.2))   

n = 1 # 1
R = 8.314 # J/molK
T = 278 #K
F = 96485.3329 # C/mol
b=2*R*T/(n*F) #V

sig_l = 9.169e-3 # S/cm
i0 = 7.2e-4 # A/cm2
S = 9.3e3 # cm2/cm3

rho_l = 1/sig_l #Ohm cm

#L0 = np.sqrt(b/(2*i0*S*rho_l))*1e4 #um

Jd = 0.01
L0 = 57

xmin = 1e-1
xmax = 3e1

ymin = 0
ymax = 350

ax.set_ylim((ymin, ymax))
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
    
    
    dftemp = dfplot.loc[dfplot['Wet_Thickness(um)']==wt, ('Thickness(um)', 'Sample')].drop_duplicates()
    
    ss_arr = np.array(list(map(int, dftemp.loc[:,'Sample'].tolist())))
    tt_arr = np.array(dftemp.loc[:,'Thickness(um)'].tolist())
    
    ttss_arr = tt_arr + ss_arr*1e-2
    #the number of thicker samples within this wet thickness
    N_Thicker = sum(ttss_arr>ttss)
    #print((ttss_arr, ttss, N_Thicker))

    #the marker is determined by how many thicker samples there are
    mark = markers[N_Thicker]
    
    return mark





for batch, df in dfplot.groupby(by = ['Batch']):

    
    ttss=np.array([])
    wtt=np.array([])
    ccc = np.array([])
    ttt = np.array([])
    
    for sample, group in df.groupby(by = ['Sample']):
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
        
        ttss = np.append(ttss, tt+int(sample)*1e-2)
        wtt = np.append(wtt, wt)
        ttt = np.append(ttt,tt)
        
        Ldiff = ld*L0
        
        Cc = np.interp(tt,np.flip(Ldiff),np.flip(cc))
        ccc = np.append(ccc,Cc)
        
        ax.plot(Cc, tt, marker=markerfun(wt,ttss[-1]), markerfacecolor=colfun(wt), markersize=4, alpha=0.5, color = 'k', zorder =100)
        #ax.hlines(tt,0.1,Cc,color = colfun(wt), linestyle = '--', zorder = 0)    
        
        #ax.plot(cc, Ldiff, marker=markerfun(wt,ttss[-1]),linestyle='-', color=colfun(wt), label=tt)

    ax.plot(ccc,ttt)
#for i, (name, group) in enumerate(dfCrit.groupby(by=['Batch'])):




Cols = ['#e36414','#0db39e','#748cab','#8a5a44','#264653','#e9c46a']
#Cols2 = ['#e76f51', '#f4a261', '#e9c46a', '#2a9d8f', '#264653']
markers = ['o','s','d', 'v', '^', '>']
lines = ['-',':','--','-.', '','-']
def stylefun(G):
    return (Cols[G], markers[G], lines[G])


#df = dfCrit.loc[dfCrit['Batch'].isin(['211202_NMC', '220203_NMC']), :].copy()

Cc = dfCrit.loc[dfCrit['Batch']=='211202_NMC', 'C-rate(prog)']
index2 = (Cc > 0.1) & (Cc < 4.8) 



for i, (name, group) in enumerate(dfCrit.groupby(by=['Batch'])):
    
    df = group

    Cc = df.loc[:, 'C-rate(prog)']


    #index2 = (Cc > 0.1) & (Cc < 4.8) 
    index = (Cc > 0.1) & (Cc < 4.8) & ~df['Thickness_lo(um)'].isnull() & ~df['Thickness_hi(um)'].isnull()
    
    index = ~df['Thickness_lo(um)'].isnull() & ~df['Thickness_hi(um)'].isnull()
    
    if name == '211202_NMC': 
        index = index & index2

    xx = np.array(df.loc[index,'C-rate_mean(1/h)'].tolist())
    yy = np.array(df.loc[index,'Thickness_max(um)'].tolist())
        
    lo = np.array(df.loc[index,'Thickness_lo(um)'].tolist())
    hi = np.array(df.loc[index,'Thickness_hi(um)'].tolist())
    ee = np.vstack([lo,hi])

    ax.errorbar(xx, yy, yerr = ee, 
                    marker = stylefun(i)[1], 
                    linestyle = stylefun(i)[2], 
                    color = stylefun(i)[0], 
                    capsize = 2, 
                    label = name, 
                    markersize=4, 
                    linewidth = 1.5, 
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


ax.set_xlabel('C-rate [1/h]')
ax.set_ylabel('Penetration depth \n[$\mu$m]')
handles, labels = ax.get_legend_handles_labels()

ax.set_xscale('log')

ax.legend(handles, labels, 
          loc = 'upper right', 
          ncol = 2)

fig.tight_layout()
plt.show()

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



dfplot = dfall.loc[dfall['Batch']=='M. Singh (2016), NMC'].copy()

fig, ax= plt.subplots(figsize=(10,6))


for name, group in dfplot.groupby(by = ['Cycle']):
    label = group['C-rate(1/h)'].unique()[0]
    group.plot(x='Thickness(um)',y='Capacity(mAh/cm2)', marker = 'o', label = label, ax = ax)


ax.set_ylabel('Capacity(mAh/cm2)')
ax.set_xlabel('Thickness(um)')
