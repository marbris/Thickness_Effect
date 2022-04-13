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


#dfall, dfCapCrit, dfOPCrit = main.dfall_dfCrit(read_json = False, write_json = True)
dfall, dfCapCrit, dfOPCrit = main.dfall_dfCrit()

    

#%% Calculating diffusion depth



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
L0 = 57

xmin = -200
xmax = 250

ymin = 0
ymax = 8

ax.set_ylim((ymin, ymax))
ax.set_xlim((xmin, xmax))


Jd=0.01
n = 1 # 1
R = 8.314 # J/molK
T = 298 #K
F = 96485.3329 # C/mol
b=2*R*T/(n*F) #V

eta_d = lambda eta0: b*np.arcsinh(Jd*np.sinh(eta0/b)) 
Ld = lambda eta0: np.log( np.tanh( eta0/(4*b) )/np.tanh( eta_d(eta0)/(4*b) ))



#OPdf.loc[:,'Penetration_Depth(x/L0)'] 

Cols = ['#e36414','#0db39e','#748cab','#8a5a44','#264653','#e9c46a']
#Cols2 = ['#e76f51', '#f4a261', '#e9c46a', '#2a9d8f', '#264653']
markers = ['o','s','d', 'v', '^', '>']
lines = ['-',':','--','-.']
def stylefun(G):
    return (Cols[np.mod(G,len(Cols))], markers[np.mod(G,len(markers))], lines[np.mod(G,len(lines))])



BatchList = ['211202_NMC']
dfplot = dfall.loc[dfall['Batch'].isin(BatchList),:]

for i, (sample, df) in enumerate(dfplot.groupby(by=['Sample'])):
    
    
    
    PenDepth = df.loc[:,'Avg_Penetration_Depth(x/L0)'].to_numpy(dtype=float)*L0
    
    #PenDepth = df.loc[:,'Avg_Overpotential(V)'].apply(Ld).to_numpy(dtype=float)*L0
    
    Thickness = df.loc[:,'Thickness(um)'].to_numpy(dtype=float)
    
    xx = (PenDepth-Thickness)
    
    yy = df.loc[:,'Avg_DCapacity(mAh/cm2)'].to_numpy(dtype=float)


    
    ax.plot(xx,yy, 
            marker = stylefun(i)[1], 
            linestyle = stylefun(i)[2], 
            color = stylefun(i)[0])
    
    
    
    i_last = np.argmax(np.isnan(xx))-1
    
    plt.text(xx[0],yy[0],'{} $\mu m$'.format(Thickness[-1]))
    

plt.vlines(0, ymin, ymax, 
           linestyle = ':', 
           color = 'k', 
           linewidth = 1, 
           zorder=0)

plt.fill_between([0,xmax],[ymax,ymax],ymin,
                 color='k',
                 alpha = 0.1, 
                 zorder = 0)

ax.set_ylabel('Areal Discharge Capacity \n[mAh/cm$^2$]')
ax.set_xlabel('Excess Penetration Depth [$\mu$m]')

fig.tight_layout()
plt.show()
    
#%%


#LegOrd = 0

for i, (batch, df) in enumerate(dfOPCrit[dfOPCrit['Batch'].isin(BatchList)].groupby(by=['Batch'])):
    
    
    #legsort.append(i+LegOrd)
    

    xx = np.array(df.loc[:,'Crit_C-rate(1/h)'].tolist())
    yy = np.array(df.loc[:,'Thickness(um)'].tolist())
    
    lo = np.array(df.loc[:,'Crit_C-rate_lo(1/h)'].tolist())
    hi = np.array(df.loc[:,'Crit_C-rate_hi(1/h)'].tolist())
    
    i_sort = np.argsort(xx)
    xx = xx[i_sort]
    yy = yy[i_sort]
    lo = lo[i_sort]
    hi = hi[i_sort]
    
    ee = np.vstack([lo,hi])

    ax.errorbar(xx, yy, xerr = ee, 
                    marker = stylefun(i)[1], 
                    linestyle = stylefun(i)[2], 
                    color = stylefun(i)[0],
                    markerfacecolor = 'w',
                    capsize = 2, 
                    label = batch, 
                    markersize=4.5, 
                    linewidth = 1, 
                    zorder = 120)



ax.set_xlabel('C-rate [1/h]')
ax.set_ylabel('Cathode Thickness [$\mu$m]')
#handles, labels = ax.get_legend_handles_labels()

ax.set_xscale('log')



#LegOrd = 50

for i in range(4):
    #legsort.append(i+LegOrd)
    ax.plot(np.zeros(1), np.zeros(1), color='w', alpha=0, label=' ')
    


handles, labels = ax.get_legend_handles_labels()

legsort = [2]*4 + [3]*2 + [4]*4 + [1]*2

handles2, labels2, legsort2 = zip(*sorted(zip(handles, labels, legsort), key=lambda k: k[2], reverse=False))

ax.legend(handles2, labels2, 
          ncol = 2, 
          framealpha = 0, 
          columnspacing=4, 
          handletextpad = 0.3, 
          labelspacing = 0.3,
          loc = 'upper right',
          bbox_to_anchor = (1.00, 0.87))

plt.text(7, 345, 'Optimal Thickness', 
         fontweight = 'bold',
         verticalalignment = 'top')
plt.text(2, 345, 'Penetration Depth \n($I = 0.01 I_0$)', 
         fontweight = 'bold',
         verticalalignment = 'top')

#ax.legend(handles, labels, 
#          loc = 'upper right', 
#          ncol = 2)

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
