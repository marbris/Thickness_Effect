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

#dfall, dfCapCrit, dfOPCrit = main.dfall_dfCrit(read_json = False, write_json = True)
dfall, dfCapCrit, dfOPCrit = main.dfall_dfCrit()

    

#%% Calculating diffusion depth

#dfplot = dfall.loc[dfall['Batch']=='211202_NMC'].copy()
#dfplot = dfall.loc[dfall['Batch']=='220203_NMC'].copy()



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


Jd=0.01
n = 1 # 1
R = 8.314 # J/molK
T = 298 #K
F = 96485.3329 # C/mol
b=2*R*T/(n*F) #V


eta0 = lambda I: b*np.arcsinh(I/(2*S*i0))
eta_d = lambda I: b*np.arcsinh(Jd*np.sinh(eta0(I)/b)) 
Ld = lambda I: np.log( np.tanh( eta0(I)/(4*b) )/np.tanh( eta_d(eta0(I))/(4*b) ))


xmin = 1e-1
xmax = 3e1

ymin = 0
ymax = 350

ax.set_ylim((ymin, ymax))
ax.set_xlim((xmin, xmax))

Cols = ['#e36414','#0db39e','#748cab','#8a5a44','#264653','#e9c46a']
#Cols2 = ['#e76f51', '#f4a261', '#e9c46a', '#2a9d8f', '#264653']
markers = ['o','s','d', 'v', '^', '>']
lines = ['-',':','--','-.', '','-']
def stylefun(G):
    return (Cols[G], markers[G], lines[G])


#df = dfCrit.loc[dfCrit['Batch'].isin(['211202_NMC', '220203_NMC']), :].copy()

Cc = dfCapCrit.loc[dfCapCrit['Batch']=='211202_NMC', 'C-rate(prog)']
index2 = (Cc > 0.1) & (Cc < 4.8) 

#legsort = []
#LegOrd = 100

for i, (batch, df) in enumerate(dfCapCrit.groupby(by=['Batch'])):

    
    #legsort.append(i+LegOrd)

    Cc = df.loc[:, 'C-rate(prog)']


    #index2 = (Cc > 0.1) & (Cc < 4.8) 
    index = (Cc > 0.1) & (Cc < 4.8) & ~df['Thickness_lo(um)'].isnull() & ~df['Thickness_hi(um)'].isnull()
    
    index = ~df['Thickness_lo(um)'].isnull() & ~df['Thickness_hi(um)'].isnull()
    
    if batch == '211202_NMC': 
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
                    label = batch, 
                    markersize=4, 
                    linewidth = 1.5, 
                    zorder = 110)




xx = np.logspace(np.log10(xmin), np.log10(xmax),1000)

ff = np.logspace(-100,100,11)

for f in ff:

    yy = Ld(xx*f)*10
    #ax.plot(xx,yy,linestyle = '-', color = 'k')








ax.set_xlabel('C-rate [1/h]')
ax.set_ylabel('Cathode Thickness [$\mu$m]')
#handles, labels = ax.get_legend_handles_labels()

ax.set_xscale('log')



handles, labels = ax.get_legend_handles_labels()

#legsort = [2]*4 + [3]*2 + [4]*4

#handles2, labels2, legsort2 = zip(*sorted(zip(handles, labels, legsort), key=lambda k: k[2], reverse=False))

ax.legend(handles, labels, 
          ncol = 2, 
          framealpha = 0, 
          columnspacing=2, 
          handletextpad = 0.3, 
          labelspacing = 0.3,
          loc = 'upper right',
          bbox_to_anchor = (1.00, 0.87))

plt.text(7, 345, 'Optimal Thickness', 
         fontweight = 'bold',
         verticalalignment = 'top')
plt.text(2.5, 345, 'Optimal C-Rate \n($I = 0.01 I_0$)', 
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

for name, group in dfall[dfall['Batch']=='211202_NMC'].groupby(by = ['Sample']):
    cc = group['Cycle'].to_numpy()
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

ax.set_xlabel('Cycle')
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




#%%



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
L0 = 100


Jd=0.01
n = 1 # 1
R = 8.314 # J/molK
T = 298 #K
F = 96485.3329 # C/mol
b=2*R*T/(n*F) #V


#getting the overpotential from butler volmer
eta0 = lambda I: b*np.arcsinh(I/(2*S*i0))



eta_d = lambda I: b*np.arcsinh(Jd*np.sinh(eta0(I)/b)) 
Ld = lambda I: np.log( np.tanh( eta0(I)/(4*b) )/np.tanh( eta_d(eta0(I))/(4*b) ))

xx = np.logspace(np.log10(xmin), np.log10(xmax),1000)

ff = np.logspace(0,10,11)

for f in ff:

    yy = Ld(xx*f)
    ax.plot(xx,yy,linestyle = '-', color = 'k')
    plt.text(xx[-1], yy[-1], 'f = {:.0e}'.format(f))
    
    

#ax.set_ylim((ymin, ymax))
ax.set_xlim((1e-1, 40))

ax.set_ylabel('$L_d/L_{\Omega}$')
ax.set_xlabel('C-Rate (1/h)')


ax.set_xscale('log')

fig.tight_layout()



#%%


#%%



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
L0 = 100


Jd=0.01
n = 1 # 1
R = 8.314 # J/molK
T = 298 #K
F = 96485.3329 # C/mol
b=2*R*T/(n*F) #V


#getting the overpotential from butler volmer
eta0 = lambda I: b*np.arcsinh(I/(2*S*i0))

eta_d = lambda I: b*np.arcsinh(Jd*np.sinh(eta0(I)/b)) 

Ld = lambda I: np.log( np.tanh( eta0(I)/(4*b) )/np.tanh( eta_d(eta0(I))/(4*b) ))

xx = [xmin, xmax]

ff = np.logspace(-100,100,2000)

yy1 = Ld(xmin*ff)
yy2 = Ld(xmax*ff)

ax.plot(ff,yy1,linestyle = '-', color = 'r', label = 'I = 0.1C')
ax.plot(ff,yy2,linestyle = '-', color = 'b', label = 'I = 30C')
    

ax.set_xscale('log')


#ax.set_ylim((ymin, ymax))
#ax.set_xlim((0, 35))

ax.set_ylabel('$L_d/L_{\Omega}$')
ax.set_xlabel('$1/2i_0S$')


fig.tight_layout()

# %%






