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
import Cycler
import simplejson as json
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


#dfall, dfCapCrit, dfOPCrit = main.dfall_dfCapCrit(read_json = False, write_json = True)
dfall, dfCapCrit, dfOPCrit = main.dfall_dfCrit()



#%% Calculating diffusion depth

#dfplot = dfall.loc[dfall['Batch']=='211202_NMC'].copy()
#dfplot = dfall.loc[dfall['Batch']=='220203_NMC'].copy()
#dfplot = dfall.loc[dfall['Batch'].isin(['211202_NMC','220203_NMC'])].copy()

BatchList = ['211202_NMC']
#Samplelist = ['05','08','12','15', '18']
Samplelist = dfall.loc[dfall['Batch']=='211202_NMC', 'Sample'].unique()

fig, ax= plt.subplots(figsize=(Col2,Col1*AR*1.2))   

Cols2 = ['k', 'r', 'b', '#2a9d8f', '#264653']
markers = ['o', 'v', 's', '^']
Cols2 = ['#e76f51', '#f4a261', '#e9c46a', '#2a9d8f', '#264653']

def colfun(i):
    return Cols2[np.mod(i,len(Cols2))]





Jd=np.exp(-1)
n = 1 # 1
R = 8.314 # J/molK
T = 298 #K
F = 96485.3329 # C/mol
b=2*R*T/(n*F) #V

eta_d = lambda eta0: b*np.arcsinh(Jd*np.sinh(eta0/b)) 
Ld = lambda eta0: np.log( np.tanh( eta0/(4*b) )/np.tanh( eta_d(eta0)/(4*b) ))


#PenDepth = df.loc[:,'Avg_Overpotential(V)'].apply(Ld).to_numpy(dtype=float)*L0

index1 = (dfall['Batch'].isin(BatchList)) & (dfall['Sample'].isin(Samplelist))

#L0 = 57
ttss = np.array([])
wtt = np.array([])
ttt = np.array([])


tt = dfall.loc[index1, 'Thickness(um)'].unique()
#plt.hlines(tt,9e-2,3e-1,linestyle = ':', color='black')

RFile='Data/McMullin/RIon_WT_220428_NMC.csv'
df_RW = pd.read_csv(RFile)



Jdlist=[0.01]
for si, (name, group) in enumerate(dfall.loc[index1, :].groupby(by = 'Sample')):

    for i, Jdi in enumerate(Jdlist):

        eta_d = lambda eta0: b*np.arcsinh(Jdi*np.sinh(eta0/b)) 
        Ld = lambda eta0: np.log( np.tanh( eta0/(4*b) )/np.tanh( eta_d(eta0)/(4*b) ))

        index = group['Cycle']<30
            
        ld = group.loc[index,'Avg_Overpotential(V)'].apply(Ld).to_numpy(dtype=float)

        Wet_Thickness = group['Wet_Thickness(um)'].unique()[0]
        L0 = Cycler.get_L0(Wet_Thickness)
        #print(L0)
        Ldiff = ld*L0
        cc = group.loc[index,'Avg_Current(mA/cm2)'].to_numpy(dtype=float)

        if name == Samplelist[0]:
            label = 'Penetration Depth, J$_d$ = {:.2}'.format(Jdi)
        else:
            label = '_No_Label_'
        
        print(i)
        ax.plot(cc, Ldiff, marker='.',linestyle='-', color=colfun(si),  label = '{} $\mu$m'.format(thickness)) #, label=label )

        thickness = group['Thickness(um)'].unique()[0]
       
        nani = np.argmax(np.isnan(Ldiff))-1
        cc = cc[:nani]
        Ldiff = Ldiff[:nani]
        
        if ~np.all(np.diff(Ldiff) > 0):
            isort = np.argsort(Ldiff)
            Ldiff = Ldiff[isort]
            cc = cc[isort]
    
        if (thickness <= max(Ldiff)) & (thickness >= min(Ldiff)):
            C_crit = np.interp(thickness,Ldiff,cc)
        else:
            C_crit = np.nan
            C_crit_hi = np.nan
            
        #ax.plot(C_crit,thickness, marker = 'o', color = colfun(i), markerfacecolor = 'w', markersize = 8)
            
        




#df = dfCapCrit.loc[dfCapCrit['Batch'].isin(['211202_NMC', '220203_NMC']), :].copy()

Cols = ['#e36414','#0db39e','#748cab','#8a5a44','#264653','#e9c46a']


index2 = (dfCapCrit['Batch'].isin(BatchList))


df = dfCapCrit.loc[index2,:].copy()

Cc = df.loc[:, 'C-rate(prog)']
#index2 = (Cc > 0.1) & (Cc < 4.8) 
index = (Cc > 0.1) & (Cc < 4.8) & ~df['Thickness_lo(um)'].isnull() & ~df['Thickness_hi(um)'].isnull()

xx = df.loc[index,'C-rate_mean(1/h)'].to_numpy(dtype=float)
yy = df.loc[index,'Thickness_max(um)'].to_numpy(dtype=float)

#calculating the current density
#first get the nominal areacap for each sample
C01Cap = np.array([dfall.loc[(dfall['Batch']=='211202_NMC') & (dfall['Cycle']==2) & (dfall['Thickness(um)']==l),'Avg_DCapacity(mAh/cm2)'].values[0] for l in yy])    

#convert c-rate to current density
xx=xx*C01Cap
#0.04222018501989293*yy



lo = df.loc[index,'Thickness_lo(um)'].to_numpy(dtype=float)
hi = df.loc[index,'Thickness_hi(um)'].to_numpy(dtype=float)
ee = np.vstack([lo,hi])

'''
ax.errorbar(xx, yy, yerr = ee, 
                marker = 'o', 
                linestyle = '-', 
                color = '#e36414', 
                capsize = 3, 
                label = 'Optimum Thickness', 
                markersize=4, 
                linewidth = 2, 
                zorder = 110)
'''




ax.set_xlabel('Discharge Current Density (mA/cm$^2$)', fontsize=15)
ax.set_ylabel('Depth into Cathode \n[$\mu$m]', fontsize = 15)
handles, labels = ax.get_legend_handles_labels()

ax.set_xscale('log')
#ax.set_xlim((9e-2, 6))

ax.legend(handles, labels, 
          loc = 'lower left', 
          ncol = 3)

fig.tight_layout()
plt.show()
#%%


fig, ax= plt.subplots(figsize=(Col2,Col1*AR*1.2))   

BatchList = ['211202_NMC']
#Samplelist = ['05','08','12','15', '18']
Samplelist = ['02', '03', '05', '06', '07', '08', '09', '12', '13', '15', '16', '17', '18', '19']


index1 = (dfall['Batch'].isin(BatchList)) & (dfall['Sample'].isin(Samplelist)) & (dfall['Cycle']<30)

tt = dfall.loc[index1, 'Thickness(um)'].unique()
plt.hlines(tt,9e-2,3e-1,linestyle = ':', color='black')


Cols2 = ['k', 'r', 'b', '#2a9d8f', '#264653']
markers = ['o', 'v', 's', '^']

def colfun(i):
    return Cols2[np.mod(i,len(Cols2))]



Jdlist=[0.01, 0.1, np.exp(-1)]
for si, (name, group) in enumerate(dfall.loc[index1, :].groupby(by = 'Sample')):

    for i, Jdi in enumerate(Jdlist):

        eta_d = lambda eta0: b*np.arcsinh(Jdi*np.sinh(eta0/b)) 
        Ld = lambda eta0: np.log( np.tanh( eta0/(4*b) )/np.tanh( eta_d(eta0)/(4*b) ))
        #
        
        cycind = group['Cycle']==2
        Qa = group.loc[cycind,'Avg_DCapacity(mAh/cm2)'].to_numpy(dtype = float)

        
        cc = group['Avg_C-rate(1/h)'].to_numpy(dtype = float)
        eta0 = b * np.arcsinh(cc*Qa/(2*i0*S))
        Ldiff = Ld(eta0)*L0
        
        if name == Samplelist[0]:
            if i == 0:
                label = 'Penetration Depth, J$_d$ = {:.2}'.format(Jdi)
            else:
                label = 'J$_d$ = {:.2}'.format(Jdi)
        else:
            label = '_No_Label_'

        ax.plot(cc,Ldiff,marker = 'o',color = colfun(i),markersize = 2, label=label)
        
        thickness = group['Thickness(um)'].unique()[0]
        
        nani = np.argmax(np.isnan(Ldiff))-1
        cc = cc[:nani]
        Ldiff = Ldiff[:nani]
        
        if ~np.all(np.diff(Ldiff) > 0):
            isort = np.argsort(Ldiff)
            Ldiff = Ldiff[isort]
            cc = cc[isort]
    
        if (thickness <= max(Ldiff)) & (thickness >= min(Ldiff)):
            C_crit = np.interp(thickness,Ldiff,cc)
        else:
            C_crit = np.nan
            C_crit_hi = np.nan
            
        ax.plot(C_crit,thickness, marker = 'o', color = colfun(i), markerfacecolor = 'w', markersize = 8, zorder = 100)
        
        

#index2 = (dfCapCrit['Batch'].isin([BatchList]))
df = dfCapCrit.loc[dfCapCrit['Batch']=='211202_NMC',:].copy()

Cc = df.loc[:, 'C-rate(prog)']
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
                label = 'Optimum Thickness', 
                markersize=4, 
                linewidth = 2, 
                zorder = 110)



ax.set_xlabel('C-rate(1/h)', fontsize=15)
ax.set_ylabel('penetration depth, \nL$_d$ [$\mu$m]', fontsize = 15)
handles, labels = ax.get_legend_handles_labels()

ax.set_xscale('log')
ax.set_xlim((9e-2, 14))
ax.set_ylim((0, 330))


ax.legend(handles, labels, 
          loc = 'upper right', 
          ncol = 4)

fig.tight_layout()
plt.show()

#%%

batch = '211202_NMC'
cycind = (dfall['Cycle']==2) & (dfall['Batch'] == batch)
Qa = dfall.loc[cycind,'Avg_DCapacity(mAh/cm2)'].to_numpy(dtype = float) #mAh/cm2
Th = dfall.loc[cycind,'Thickness(um)'].to_numpy(dtype = float)*1e-4 #cm
Rc = dfall.loc[cycind,'Avg_C-rate(1/h)'].to_numpy(dtype = float) #1/h
Qv = Qa/Th

n = 1 # 1
R = 8.314 # J/molK
T = 298 #K
F = 96485.3329 # C/mol
b=2*R*T/(n*F) #V

sig_l = 9.169e-3 # S/cm
i0 = 7.2e-4 # A/cm2
S = 9.3e3 # cm2/cm3
rho_l = 1/sig_l #Ohm cm

Jd=0.15
L0=57

eta_d = lambda eta0: b*np.arcsinh(Jd*np.sinh(eta0/b)) 
Ld = lambda eta0: np.log( np.tanh( eta0/(4*b) )/np.tanh( eta_d(eta0)/(4*b) ))

Cols = ['#e36414','#0db39e','#748cab','#8a5a44','#264653','#e9c46a']
markers = ['o', 'v', 's', '^']

def colfun(i):
    return Cols[np.mod(i,len(Cols))]

fig, ax= plt.subplots(figsize=(13,6))

#batchind = (dfall['Batch'] == '211202_NMC') & (dfall['Cycle'] <= 29)

batchind = (dfall['Batch'] == batch) & (dfall['Cycle'] <= 29)
df = dfall.loc[batchind,:].copy()

for i, (sample, group) in enumerate(df.groupby(by = 'Sample')):
    
    cycind = group['Cycle']==2
    Qa = group.loc[cycind,'Avg_DCapacity(mAh/cm2)'].to_numpy(dtype = float)

    Th = group['Thickness(um)'].unique()[0]*1e-4 #cm
    cc = group['Avg_C-rate(1/h)'].to_numpy(dtype = float)
    eta0 = b * np.arcsinh(cc*Qa/(2*i0*S))
    Ldiff = Ld(eta0)*L0
    
    
    ax.plot(cc,Ldiff,marker = 'o',color = colfun(i),markersize = 2)


    
    nani = np.argmax(np.isnan(Ldiff))-1
    cc = cc[:nani]
    Ldiff = Ldiff[:nani]

    if ~np.all(np.diff(Ldiff) > 0):
        isort = np.argsort(Ldiff)
        Ldiff = Ldiff[isort]
        cc = cc[isort]

    thickness = Th*1e4
    if (thickness <= max(Ldiff)) & (thickness >= min(Ldiff)):
        C_crit = np.interp(thickness,Ldiff,cc)
    else:
        C_crit = np.nan
        C_crit_hi = np.nan
        
    ax.plot(C_crit,thickness, marker = 'o', color = colfun(i), markerfacecolor = 'w', markersize = 8, zorder = 100)
    
    plt.hlines(thickness,1e-1,1e0,linestyle=':', color = colfun(i))
    

ax.set_xscale('log')



Cols = ['#e36414','#0db39e','#748cab','#8a5a44','#264653','#e9c46a']

index2 = (dfCapCrit['Batch'].isin([batch]))


df = dfCapCrit.loc[index2,:].copy()

Cc = df.loc[:, 'C-rate(prog)']
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
                label = 'Optimum Thickness', 
                markersize=4, 
                linewidth = 2, 
                zorder = 110)



#p = np.polyfit(Th,eta0,1)

#plt.plot(Th*1e4, eta0, marker = 'o', linestyle = '')
#plt.plot([min(Th*1e4),max(Th*1e4)], np.polyval(p,[min(Th),max(Th)]),color = 'r')






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
    group.plot(x='Thickness(um)',y='Discharge_Capacity(mAh/cm2)', marker = 'o', label = label, ax = ax)


ax.set_ylabel('Capacity(mAh/cm2)')
ax.set_xlabel('Thickness(um)')




#%%


fig, ax= plt.subplots(figsize=(Col2,Col1*AR*1.2))  

ax.set_xscale('log')



Cols = ['#e36414','#0db39e','#748cab','#8a5a44','#264653','#e9c46a']

index2 = (dfCapCrit['Batch'].isin([batch]))


df = dfCapCrit.loc[index2,:].copy()

Cc = df.loc[:, 'C-rate(prog)']
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
                label = 'Optimum Thickness', 
                markersize=4, 
                linewidth = 2, 
                zorder = 110)


n = 1 # 1
R = 8.314 # J/molK
T = 298 #K
F = 96485.3329 # C/mol
b=2*R*T/(n*F) #V

sig_l = 9.169e-3 # S/cm
i0 = 7.2e-4 # A/cm2
S = 9.3e3 # cm2/cm3
rho_l = 1/sig_l #Ohm cm

ll = (R*T/(n*F*i0*S*rho_l))**(1/2)*1e4

Cols2 = ['k', 'r', 'b']

def colfun(i):
    return Cols2[np.mod(i,len(Cols2))]


Jdlist=[0.01, 0.1, np.exp(-1)]
for i, Jdi in enumerate(Jdlist):
    label = 'Penetration Depth, J$_d$ = {:.2}'.format(Jdi)
    plt.hlines(ll*np.arctanh(1-Jdi), 9e-2, 14, linestyle = ':', color = colfun(i), label = label)


ax.set_xlabel('C-rate(1/h)', fontsize=15)
ax.set_ylabel('penetration depth \nL$_d$ [$\mu$m]', fontsize = 15)
handles, labels = ax.get_legend_handles_labels()

ax.set_xscale('log')
ax.set_xlim((9e-2, 14))
ax.set_ylim((0, 200))


ax.legend(handles, labels, 
          loc = 'lower left', 
          ncol = 1, 
          framealpha = 0.9, 
          columnspacing=1, 
          handletextpad = 0.3, 
          labelspacing = 0.3)

fig.tight_layout()
plt.show()



#%%


fig, ax= plt.subplots(figsize=(Col2,Col1*AR*1.2))  


n = 1 # 1
R = 8.314 # J/molK
T = 298 #K
F = 96485.3329 # C/mol
b=2*R*T/(n*F) #V

sig_l = 9.169e-3 # S/cm
i0 = 7.2e-4 # A/cm2
S = 9.3e3 # cm2/cm3
rho_l = 1/sig_l #Ohm cm

ll = (R*T/(n*F*i0*S*rho_l))**(1/2)*1e4

index2 = (dfCapCrit['Batch'].isin([batch]))
df = dfCapCrit.loc[index2,:].copy()

Cc = df.loc[:, 'C-rate(prog)']
#index2 = (Cc > 0.1) & (Cc < 4.8) 
index = (Cc > 0.1) & (Cc < 4.8) & ~df['Thickness_lo(um)'].isnull() & ~df['Thickness_hi(um)'].isnull()

xx = df.loc[index,'C-rate_mean(1/h)'].to_numpy(dtype=float)
yy = df.loc[index,'Thickness_max(um)'].to_numpy(dtype=float)
    
lo = df.loc[index,'Thickness_lo(um)'].to_numpy(dtype=float)
hi = df.loc[index,'Thickness_hi(um)'].to_numpy(dtype=float)

jj = 1-np.tanh(yy/ll)
#since a larger thickness corresponds to a lower Jd, the error bars are flipped
jj_hi = abs(jj - (1-np.tanh((yy-lo)/ll)))
jj_lo = abs(jj - (1-np.tanh((yy+hi)/ll)))


ee = np.vstack([jj_lo,jj_hi])

ax.errorbar(xx, jj, yerr = ee, 
                marker = 'o', 
                linestyle = '-', 
                color = '#e36414', 
                capsize = 3, 
                label = 'Fractional current at Optimum Thickness', 
                markersize=4, 
                linewidth = 2, 
                zorder = 110)



ax.set_xlabel('C-rate(1/h)', fontsize=15)
ax.set_ylabel('Fractional current \nJ$_d$', fontsize = 15)
handles, labels = ax.get_legend_handles_labels()

ax.set_xscale('log')
ax.set_xlim((9e-2, 14))
#ax.set_ylim((0, 330))


ax.legend(handles, labels, 
          loc = 'upper left', 
          ncol = 1)

fig.tight_layout()
plt.show()

