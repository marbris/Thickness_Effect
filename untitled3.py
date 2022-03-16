#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 10:44:23 2022

@author: martin
"""

from matplotlib import pyplot as plt

#import re
import pandas as pd
import Cycler
import numpy as np
#import simplejson as json

SMALL_SIZE = 15
MEDIUM_SIZE = 15
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


#%%

Cols = ['black', '#219ebc', '#e36414', '#606c38', 'green']

prog_cols = {"Martin_cycles_1": Cols[1],
             "Martin_cycles_2": Cols[2],
             "Martin_cycles_3": Cols[3]}

fig, ax= plt.subplots(figsize=(13,6))

ymin = -0.2
ymax = 9

xmin = 9e-2
xmax = 3e1

ax.set_ylim((ymin, ymax))
#ax.set_ylim((0, 250))
ax.set_xlim((1e-1, 2e2))
#ax.set_xlim((xmin, xmax))

#ax.fill_between([xmin, 3e-1], [ymax, ymax], y2 = ymin, color = Cols[0], alpha = 0.2)
#ax.fill_between([0.3, 1.1], [ymax, ymax], y2 = ymin, color = Cols[1], alpha = 0.2)
#ax.fill_between([1.1, 3.9], [ymax, ymax], y2 = ymin, color = Cols[2], alpha = 0.2)
#ax.fill_between([3.9, 14.1], [ymax, ymax], y2 = ymin, color = Cols[3], alpha = 0.2)

colfun = lambda x: [x, 0.5, x] 

Cols2 = ['#e76f51', '#e9c46a', '#2a9d8f', '#264653']
def colfun(t):
    if t>140:
        col = Cols2[0]
    elif t>100:
        col = Cols2[1]
    elif t>70:
        col = Cols2[2]
    else:
        col = Cols2[3]
    
    return col
        
        

DataDirectory = "Data/"
BatchLabel = '211202_NMC'

SL = Cycler.get_SampleList(DataDirectory)



#regexp=r"CoinCell_(\d\d)_211202_NMC_StatisticByStep.CSV$"

tt=[]
pp=[]
diameter = 1.23 #cm
area = (diameter/2)**2*3.14 #electrode area, cm2

dfall = pd.DataFrame()

PlotList = SL[BatchLabel]
PlotList.reverse()

for SampleID in PlotList:
    
    SampleInfo, BatchInfo = Cycler.get_SampleInfo(BatchLabel, SampleID, DataDirectory)
    
    Thickness = SampleInfo['ECCthickness'] - BatchInfo['CurrentCollector']['Thickness'] #micrometer
    tt.append(Thickness)
    Prog = SampleInfo['Cycler_Program']
    Mass = (SampleInfo['ECCmass']-BatchInfo['CurrentCollector']['Mass'])*1e-3 #grams
    
    Slurry_Mass = BatchInfo['Slurry']['AM']['Mass'] + BatchInfo['Slurry']['Binder']['Mass']*BatchInfo['Slurry']['Binder']['Binder_Concentration'] + BatchInfo['Slurry']['Carbon']['Mass']
    
    AM_Mass_frac = BatchInfo['Slurry']['AM']['Mass']/Slurry_Mass
    AM_Mass = Mass * AM_Mass_frac

    df = Cycler.get_CycleData(BatchLabel, SampleID, DataDirectory, Properties=['Cycle_ID', 'Discharge_Current(mA)', 'Discharge_Capacity(mAh)'])

    df.loc[:,'Current(mA/cm2)'] = -df.loc[:,'Discharge_Current(mA)']/area
    df.loc[:,'Capacity(mAh/cm2)'] = df.loc[:,'Discharge_Capacity(mAh)']/area
    df.loc[:,'Capacity(mAh/gAM)'] = df.loc[:,'Discharge_Capacity(mAh)']/AM_Mass
    
    
    
    df.loc[:,'C-rate(1/h)'] = df.loc[:,'Current(mA/cm2)']/df.loc[df['Cycle_ID']==2,'Capacity(mAh/cm2)'].values[0]
    
    #I'm dropping the final few cycles, since those repeat the low-C cycles.
    df.drop(df.loc[(df['C-rate(1/h)']<0.3) & (df['Cycle_ID']>2)].index, inplace=True)

    df.loc[:,'Thickness(um)'] = Thickness
    df.loc[:,'Cycler_Program'] = int(SampleInfo['Cycler_Program'][-1:])
    pp.append(int(SampleInfo['Cycler_Program'][-1:]))
    dfall = dfall.append(df, ignore_index=True)

    
    #df.plot(x = 'C-rate(1/h)', y = 'Capacity(mAh/cm2)', ax=ax, color = colfun(Thickness), marker = 'o', logx=True, markersize = Thickness*0.06, label = '{:.0f} $\mu m$'.format(Thickness))
    #df.plot(x = 'Current(mA/cm2)', y = 'Capacity(mAh/gAM)', ax=ax, color = colfun(Thickness/200), marker = 'o', logx=True, markersize = Thickness*0.06, label = '{:.0f} $\mu m$'.format(Thickness))
    #df.plot(x = 'C-rate(1/h)', y ='Capacity(mAh/gAM)', ax=ax, color = colfun(Thickness), marker = 'o', logx=True, markersize = Thickness*0.06, label = '{:.0f} $\mu m$'.format(Thickness))
    df.plot(x = 'Current(mA/cm2)', y = 'Capacity(mAh/cm2)', ax=ax, color = colfun(Thickness), marker = 'o', logx=True, markersize = Thickness*0.06, label = '{:.0f} $\mu m$'.format(Thickness))
    

#this is a dummy line just to align the legend.
#ax.plot(np.zeros(1), np.zeros([1,2]), color='w', alpha=0, label=' ')
#pp.append(3)
#tt.append(0)

#this is a dummy line just to align the legend.
#for i in range(3):
#    ax.plot(np.zeros(1), np.zeros([1,2]), color='w', alpha=0, label=' ')
#    pp.append(1)
#    tt.append(0)


handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
#labels, handles, tt2, pp2 = zip(*sorted(zip(labels, handles, tt, pp), key=lambda k: (np.mod(k[3]+1,3), k[2]), reverse=True))
labels, handles, tt2 = zip(*sorted(zip(labels, handles, tt), key=lambda k: k[2], reverse=True))
ax.legend(handles, labels, ncol = 1, loc='upper right', framealpha = 0, fontsize = 13)

#handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
#labels, handles, tt2 = zip(*sorted(zip(labels, handles, tt), key=lambda t: t[2], reverse=True))
#ax.legend(handles, labels, ncol = 1, loc='upper right')



#ax.set_ylabel('Discharge Capacity [mAh/g$_{AM}$]')#, fontsize = 15)
ax.set_ylabel('Discharge Capacity [mAh/cm$^2$]', fontsize = 15)
ax.set_xlabel('Current Density [mA/cm$^2$]', fontsize = 15)
#ax.set_xlabel('C-rate [h$^{-1}$]')



#%%



#Crates = [0.01, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.3, 1.6, 1.8, 2.1, 2.4, 2.7, 3.2, 3.6, 4.2, 4.8, 5.6, 6.4, 7.4, 8.5, 9.8, 11.3, 13.1, 15.1, 17.4, 20]

Crates = [ [0.1, 0.5, 0.8, 1.2, 1.8, 2.7, 4.2, 6.4, 9.8, 15.1],
           [0.1, 0.6, 0.9, 1.3, 2.1, 3.2, 4.8, 7.4, 11.3, 17.4],
           [0.1, 0.7, 1, 1.6, 2.4, 3.6, 5.6, 8.5, 13.1, 20] ]


for i in range(3):
    for cr in Crates[i]:
        
        dfall.loc[(abs(dfall.loc[:,'C-rate(1/h)']/cr-1)<=0.18) & (dfall.loc[:,'Cycler_Program'] == i+1),'C-rate(rnd)'] = cr
        dfall.loc[abs(dfall.loc[:,'C-rate(1/h)']/cr-1)<=0.18,'C-rate(rnd-frac)'] = dfall.loc[:,'C-rate(1/h)']/cr-1
        
        thicks = np.sort(dfall.loc[dfall['C-rate(rnd)'] == cr,'Thickness(um)'].unique())
        
        for tt in thicks:
            dfall.loc[(dfall['C-rate(rnd)'] == cr) & (dfall['Thickness(um)'] == tt),'Avg_DCapacity(mAh/cm2)'] = dfall.loc[(dfall['C-rate(rnd)'] == cr) & (dfall['Thickness(um)'] == tt),'Capacity(mAh/cm2)'].mean()
        
        
    #dfall.loc[dfall['Cycler_Program'] == i+1].plot(x = 'C-rate', y = 'C-rate(rnd)', ax = ax, marker = 'o', linestyle = '', logx=True, logy=True, color = Cols[i])

#%%    

#dfall.plot(x = 'C-rate(rnd)', y = 'C-rate(rnd)', ax = ax, marker = 'o', linestyle = '', logx=True, logy=True, color = 'black')



#dfall.plot(x = 'C-rate', y = 'C-rate(rnd)', ax = ax, marker = 'o', linestyle = '', logx=True, logy=True)



fig, ax= plt.subplots(figsize=(13,6))

ax.set_ylim((-0.5, 8.5))
ax.set_xlim((0, 200))

Cols = ['black', '#219ebc', '#e36414', '#606c38', 'green']

crts = np.sort(dfall.loc[:,'C-rate(rnd)'].unique())

crts2 = crts[crts<=13.1]

#crts2 = crts[(crts>0.1) & (crts<=1.0)]
#crts2 = crts[(crts>1.0) & (crts<=3.6)]
#crts2 = crts[(crts>3.6) & (crts<=13.1)]

#crts2 = np.insert(crts2,0,0.1)

Shroud = False

tc = []
tce = np.empty((2,0))
Cc = []
mcap = []

tc_2 = []
tce_2 = np.empty((2,0))
Cc_2 = []

tc_3 = []
tce_3 = np.empty((2,0))
Cc_3 = []

tc_4 = []
tce_4 = np.empty((2,0))
Cc_4 = []

#Im using these to sort the legend
gg=[]
cc=[]

for cr in crts2:
    df3 = pd.DataFrame()
    df3 = dfall[dfall['C-rate(rnd)'] == cr]
    df3.sort_values(by = ['Thickness(um)'], ignore_index=True, inplace = True)
    df3tt = df3['Thickness(um)'].unique()
    
    if cr == 0.1:
        color = Cols[0]
        ms = 1
        Shroud = False
        gg.append(0)
    elif cr <= 1.0:
        color = Cols[1]
        ms = 3 + (cr-0.5)/(1.0-0.5)*9
        Shroud = False
        gg.append(1)
    elif cr <= 3.6:
        color = Cols[2]
        ms = 3 + (cr-1.2)/(3.6-1.2)*9
        Shroud = True
        gg.append(2)
    elif cr <= 13.1:
        color = Cols[3]
        ms = 3 + (cr-4.2)/(13.1-4.2)*9
        Shroud = False
        gg.append(3)
    
    cc.append(cr)    
    #markersize = np.log10(cr)*6+5
    maxCap = df3.loc[df3['Capacity(mAh/cm2)'] == df3['Capacity(mAh/cm2)'].max(),'Avg_DCapacity(mAh/cm2)'].values[0]
    maxCapTh = df3.loc[df3['Avg_DCapacity(mAh/cm2)']==maxCap, 'Thickness(um)'].values[0]
    
    if (maxCapTh != df3.loc[:,'Thickness(um)'].max()) & Shroud:
        ax.plot(maxCapTh, maxCap, marker = 'o', markersize = ms*2 + 10, color = color, alpha = 0.3)
        mcap.append(maxCap)
        tc.append(maxCapTh)
        Cc.append(cr)
        
        #calculating error bar of the critical thickness. taking halfway to adjacent points
        temp = (df3tt[df3tt != maxCapTh] - maxCapTh)/2
        etemp = np.array([[-temp[temp<0].max()], [temp[temp>0].min()]])
        tce = np.hstack([tce,etemp])
        #print(temp)
        
        
    elif cr <= 1:
        maxCapTh = df3.loc[:,'Thickness(um)'].max()
        tc_2.append(maxCapTh)
        Cc_2.append(cr)
        
        temp = (df3tt[df3tt != maxCapTh] - maxCapTh)/2
        etemp = np.array([[-temp[temp<0].max()], [200-maxCapTh]])
        tce_2 = np.hstack([tce_2,etemp])
    
    elif cr >= 4.2:
        tc_3.append(maxCapTh)
        Cc_3.append(cr)
        
        temp = (df3tt[df3tt != maxCapTh] - maxCapTh)/2
        etemp = np.array([[maxCapTh-40], [temp[temp>0].min()]])
        tce_3 = np.hstack([tce_3,etemp])
        
        
    
    df3.plot(x = 'Thickness(um)', y = 'Avg_DCapacity(mAh/cm2)', ax = ax, marker = 'o', markersize = ms, color = color, label = str(cr))#'C: {}'.format(cr))


    

#this is a dummy line just to align the legend.
for i in range(3):
    ax.plot(np.zeros(1), np.zeros([1,2]), color='w', alpha=0, label=' ')
    gg.append(1)
    cc.append(0)

#this is a dummy line just to align the legend.
for i in range(8):
    ax.plot(np.zeros(1), np.zeros([1,2]), color='w', alpha=0, label=' ')
    gg.append(0)
    cc.append(0)    

handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
labels, handles, tt2, pp2 = zip(*sorted(zip(labels, handles, cc, gg), key=lambda k: (k[3], k[2]), reverse=True))
ax.legend(handles, labels, ncol = 4, framealpha = 0, bbox_to_anchor=(0,0.93), loc = 'upper left')

ax.text(12, 8, 'C-rates', color = 'k', fontsize = 18, horizontalalignment = 'center', verticalalignment = 'center')

#handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles, labels, ncol = 3, fontsize = 13)

ax.set_ylabel('Areal Discharge Capacity [mAh/cm$^2$]', fontsize = 15)
ax.set_xlabel('Thickness [$\mu m$]', fontsize = 15)



#%%

#C-rate vs Thickness vs Discharge Capacity

fig, ax= plt.subplots(figsize=(13,6))
crts = np.sort(dfall.loc[:,'C-rate(rnd)'].unique())

colfun = lambda x: [x, 0, 0] 

for cr in crts:
    thicks = np.sort(dfall.loc[dfall['C-rate(rnd)'] == cr,'Thickness(um)'].unique())
    
    for it in thicks:
        
        cap = dfall.loc[(dfall['C-rate(rnd)'] == cr) & (dfall['Thickness(um)'] == it),'Avg_DCapacity(mAh/cm2)'].mean()
        plt.plot(cr,it,markersize=cap*3+2, marker = 'o', color = colfun(cap/8.2))
        
    
plt.xscale('log')    
    
ax.set_ylabel('Thickness [$\mu$m]', fontsize = 15)
ax.set_xlabel('C-rate [h$^{-1}$]', fontsize = 15)





#%%

#This is the Critical Thickness vs C-rate

fig, ax= plt.subplots(figsize=(13,6))

ax.set_ylim((30, 220))
ax.set_xlim((9e-2, 14))

ms = 10
lw = 2

ax.errorbar(Cc, tc, yerr = tce, marker = 'o', linestyle = '-', color = Cols[2], capsize = 5, markersize=ms, linewidth = lw)

ax.errorbar(Cc_2[1:], tc_2[1:], yerr = tce_2[:,1:], lolims=True, marker = '', linestyle = 'none', color = Cols[1], capsize = 5, markersize=ms, linewidth = lw)
ax.errorbar(Cc_2[1:], tc_2[1:], yerr = tce_2[:,1:], marker = 'o', linestyle = 'none', color = Cols[1], capsize = 5, markersize=ms, linewidth = lw)

tce_4 = np.array([[tce_2[0,0]], [tce_2[1,0]+10]])
ax.errorbar(Cc_2[0], tc_2[0], yerr = tce_4, lolims=True, marker = '', linestyle = 'none', color = Cols[0], capsize = 5, markersize=ms, linewidth = lw)
ax.errorbar(Cc_2[0], tc_2[0], yerr = tce_4, marker = 'o', linestyle = 'none', color = Cols[0], capsize = 5, markersize=ms, linewidth = lw)

ax.errorbar(Cc_3, tc_3, yerr = tce_3, uplims=True, marker = '', linestyle = 'none', color = Cols[3], capsize = 5, markersize=ms, linewidth = lw)
ax.errorbar(Cc_3, tc_3, yerr = tce_3, marker = 'o', linestyle = 'none', color = Cols[3], capsize = 5, markersize=ms, linewidth = lw)

plt.xscale('log')

ax.set_ylabel('Critical Thickness [$\mu m$]', fontsize = 15)
ax.set_xlabel('C-rate [h$^{-1}$]', fontsize = 15)

#%%

#This is the maximum capacity vs C-rate

fig, ax= plt.subplots(figsize=(9,5))

ax.set_ylim((0, 6))
ax.set_xlim((0, 3.7))

ax.plot(Cc, mcap, marker = 'o', linestyle = '-', color = Cols[2])


ax.set_ylabel('Maximum Capacity [mAh/cm$^2$]', fontsize = 15)
ax.set_xlabel('C-rate [h$^{-1}$]', fontsize = 15)




#%%

#Here I add Zheng and Denis LFP Data

#filename='/home/martin/Documents/Research/Batteries/thickness effect/LFP_Data.csv'
filename='/home/martin/Documents/Research/Batteries/thickness effect/DYW_ZH_Data.csv'
D = pd.read_csv(filename)


#calculate area capacity
D['Qa[mAh/cm2]'] = 0
D.loc[D['Source'] == 'H. Zheng (2012)', 'Qa[mAh/cm2]'] = D['Q']
rho=2.2
D.loc[D['Source'] == 'D.Y.W. Yu (2006)', 'Qa[mAh/cm2]'] = D['Q']*rho*(D['t[mu]']*1e-4)
  


#%%

fig, ax= plt.subplots(figsize=(9,5))

ax.set_ylim((0, 4.1))
ax.set_xlim((0, 125))

dd=D[D['Source']== 'D.Y.W. Yu (2006)']

Z=dd.pivot_table(index='C', columns='t[mu]', values = 'Qa[mAh/cm2]').values
t_unique = np.sort(dd['t[mu]'].unique())
C_unique = np.sort(dd['C'].unique())

tc_DYW = []
tce_DYW = np.empty((2,0))
Cc_DYW = []
mcap_DYW = []

for i, z in enumerate(Z):
    plt.plot(t_unique, z, label='C = {}'.format(C_unique[i]), marker = 'o')

    
    #markersize = np.log10(cr)*6+5
    maxCap = max(z[~np.isnan(z)])
    maxCapTh = t_unique[z == maxCap][0]
    
    if (maxCapTh != max(t_unique[~np.isnan(z)])) & (maxCapTh != min(t_unique[~np.isnan(z)])):
        ax.plot(maxCapTh, maxCap, marker = 'o', markersize = 20, color = 'red', alpha = 0.3)
        mcap_DYW.append(maxCap)
        tc_DYW.append(maxCapTh)
        Cc_DYW.append(C_unique[i])
        
        #calculating error bar of the critical thickness. taking halfway to adjacent points
        temp = (t_unique[t_unique != maxCapTh] - maxCapTh)/2
        etemp = np.array([[-temp[temp<0].max()], [temp[temp>0].min()]])
        tce_DYW = np.hstack([tce_DYW,etemp])
        #print(temp)

ax.set_ylabel('Q/A [mAh/cm$^2$]', fontsize = 15)
ax.set_xlabel('Thickness [$\mu m$]', fontsize = 15)

plt.legend(loc='upper left', ncol=3)

#%%

fig, ax= plt.subplots(figsize=(9,5))

ax.set_ylim((0, 3))
ax.set_xlim((0, 125))

dd=D[(D['Source']== 'H. Zheng (2012)') & (D['Cathode'] == 'LFP') ]

Z=dd.pivot_table(index='C', columns='t[mu]', values = 'Qa[mAh/cm2]').values
t_unique = np.sort(dd['t[mu]'].unique())
C_unique = np.sort(dd['C'].unique())

tc_HZ_LFP = []
tce_HZ_LFP = np.empty((2,0))
Cc_HZ_LFP = []
mcap_HZ_LFP = []

for i, z in enumerate(Z):
    plt.plot(t_unique, z, label='C = {}'.format(C_unique[i]), marker = 'o')

    
    #markersize = np.log10(cr)*6+5
    maxCap = max(z[~np.isnan(z)])
    maxCapTh = t_unique[z == maxCap][0]
    
    if (maxCapTh != max(t_unique[~np.isnan(z)])) & (maxCapTh != min(t_unique[~np.isnan(z)])):
        ax.plot(maxCapTh, maxCap, marker = 'o', markersize = 20, color = 'red', alpha = 0.3)
        mcap_HZ_LFP.append(maxCap)
        tc_HZ_LFP.append(maxCapTh)
        Cc_HZ_LFP.append(C_unique[i])
        
        #calculating error bar of the critical thickness. taking halfway to adjacent points
        temp = (t_unique[t_unique != maxCapTh] - maxCapTh)/2
        etemp = np.array([[-temp[temp<0].max()], [temp[temp>0].min()]])
        tce_HZ_LFP = np.hstack([tce_HZ_LFP,etemp])
        #print(temp)

ax.set_ylabel('Q/A [mAh/cm$^2$]', fontsize = 15)
ax.set_xlabel('Thickness [$\mu m$]', fontsize = 15)

plt.legend(loc='upper left', ncol=3)

#%%

fig, ax= plt.subplots(figsize=(9,5))

ax.set_ylim((0, 4))
ax.set_xlim((0, 125))

dd=D[(D['Source']== 'H. Zheng (2012)') & (D['Cathode'] == 'NMC') ]

Z=dd.pivot_table(index='C', columns='t[mu]', values = 'Qa[mAh/cm2]').values
t_unique = np.sort(dd['t[mu]'].unique())
C_unique = np.sort(dd['C'].unique())

tc_HZ_NMC = []
tce_HZ_NMC = np.empty((2,0))
Cc_HZ_NMC = []
mcap_HZ_NMC = []

for i, z in enumerate(Z):
    plt.plot(t_unique, z, label='C = {}'.format(C_unique[i]), marker = 'o')

    
    #markersize = np.log10(cr)*6+5
    maxCap = max(z[~np.isnan(z)])
    maxCapTh = t_unique[z == maxCap][0]
    
    if (maxCapTh != max(t_unique[~np.isnan(z)])) & (maxCapTh != min(t_unique[~np.isnan(z)])):
        ax.plot(maxCapTh, maxCap, marker = 'o', markersize = 20, color = 'red', alpha = 0.3)
        mcap_HZ_NMC.append(maxCap)
        tc_HZ_NMC.append(maxCapTh)
        Cc_HZ_NMC.append(C_unique[i])
        
        #calculating error bar of the critical thickness. taking halfway to adjacent points
        temp = (t_unique[t_unique != maxCapTh] - maxCapTh)/2
        etemp = np.array([[-temp[temp<0].max()], [temp[temp>0].min()]])
        tce_HZ_NMC = np.hstack([tce_HZ_NMC,etemp])
        #print(temp)

ax.set_ylabel('Q/A [mAh/cm$^2$]', fontsize = 15)
ax.set_xlabel('Thickness [$\mu m$]', fontsize = 15)

plt.legend(loc='upper left', ncol=3)


#%%

#This is the Critical Thickness vs C-rate, including DYW and HZ

fig, ax= plt.subplots(figsize=(13,6))

Lfun = lambda R, k: np.sqrt(k/R)
Rfun = lambda L, k: k/L**2 

xmin = 1e-1
xmax = 2.1e1
ymin = 0
ymax = 200

ax.set_ylim((ymin, ymax))
ax.set_xlim((xmin, xmax))

xx = np.logspace(np.log10(xmin),np.log10(xmax), 100)
#kkcms = np.array([0.3, 2, 8, 30])*1e-8
kkcms = np.array([2**i for i in range(-1,8,2)])*1e-8
kkmuh = kkcms*3.6e11

for i, k in enumerate(kkmuh):
    ax.plot(xx, Lfun(xx,k), linestyle=':', color=[0.5,0.5,0.5])
    
    if Rfun(ymax, k) < xmin:
        xpos = xmin*1.05
        ypos = Lfun(xpos, k)-12
    else:
        ypos = ymax-2
        xpos = Rfun(ymax, k)*0.87
        
    #print((xpos,ypos))
    if i != 0:
        #ypos = 200
        #xpos = Rfun(ypos, k)*0.87
        ax.text(xpos, ypos, "{:.0f}".format(kkcms[i]/1e-8), color = 'k', fontsize = 15, horizontalalignment = 'left', verticalalignment = 'top', rotation = -50)
    else:
        #xpos = 1.1e-1
        #ypos = Lfun(xpos, k)-10
        ax.text(xpos, ypos, "RL$_d^2$ = {}".format(kkcms[i]/1e-8) +  " [$10^{-8}$ cm$^2$/s]", color = 'k', fontsize = 15, horizontalalignment = 'left', verticalalignment = 'top', rotation = -30)
        

ms = 10
lw = 2

ax.errorbar(Cc, tc, yerr = tce, marker = 'o', linestyle = '-', color = Cols[2], capsize = 5, label = 'Current Work, NMC', markersize=ms, linewidth = lw)
ax.errorbar(Cc_HZ_NMC, tc_HZ_NMC, yerr = tce_HZ_NMC, marker = 's', linestyle = '', color = '#0db39e', capsize = 5, label = 'H. Zheng (2012), NMC', markersize=ms, linewidth = lw)
ax.errorbar(Cc_HZ_LFP, tc_HZ_LFP, yerr = tce_HZ_LFP, marker = 'd', linestyle = '--', color = '#748cab', capsize = 5, label = 'H. Zheng (2012), LFP', markersize=ms, linewidth = lw)
ax.errorbar(Cc_DYW, tc_DYW, yerr = tce_DYW, marker = 'v', linestyle = ':', color = '#8a5a44', capsize = 5, label = 'D.Y.W. Yu (2006), LFP', markersize=ms, linewidth = lw)

ax.set_ylabel('Penetration Depth, L$_d$ [$\mu m$]', fontsize = 15)
ax.set_xlabel('C-rate, R [h$^{-1}$]', fontsize = 15)

ax.set_xscale('log')

# #setting second x-axis
# R2rl = lambda R: R*ymax**2/3.6e11*1e8
# rl2R = lambda rl: rl/(ymax**2)*3.6e11*1e-8

# secaxx = ax.secondary_xaxis('top', functions=(R2rl, rl2R))

# secaxx.set_xlabel('RL$_d^2$ [$10^{-8}$ cm$^2$/s]')

# Xtcks=kkcms[kkcms>R2rl(xmin)*1e-8]*1e8
# Xtcks_str = ['{:.0f}'.format(i) for i in Xtcks]

# secaxx.set_xscale('log')
# secaxx.get_xaxis().get_major_formatter().labelOnlyBase = False
# secaxx.set_xticks(Xtcks, minor=False)
# secaxx.set_xticks([], minor=True)
# secaxx.set_xticklabels(Xtcks_str)

# #setting second y-axis
# L2rl = lambda L: L**2*xmax/3.6e11*1e8
# rl2L = lambda rl: np.sqrt(rl/xmax*3.6e11*1e-8)

# secaxy = ax.secondary_yaxis('right', functions=(L2rl, rl2L))

# secaxy.set_ylabel('RL$_d^2$ [$10^{-8}$ cm$^2$/s]')

# Ytcks=kkcms*1e8
# Ytcks_str = ['{:.0f}'.format(i) for i in Ytcks]
# Ytcks_str[0] = '0.5'
# #secaxy.set_yscale('linear')
# secaxy.set_yticks(Ytcks, minor=False)
# #secaxy.set_xticks([], minor=True)
# secaxy.set_yticklabels(Ytcks_str)


plt.legend(loc='lower left', fontsize = 12)

plt.show()


#%%

#This is the maximum capacity vs C-rate, including DYW and HZ

fig, ax= plt.subplots(figsize=(9,5))

ax.set_ylim((0, 6))
ax.set_xlim((0, 21))

#ax.plot(Cc, mcap, marker = 'o', linestyle = '-', color = Cols[2], label = 'Current Work, NMC')
#ax.plot(Cc_DYW, mcap_DYW, marker = 'v', linestyle = ':', color = '#8a5a44', label = 'D.Y.W. Yu (2006), LFP')
#ax.plot(Cc_HZ, mcap_HZ, marker = 'd', linestyle = '--', color = '#748cab', label = 'H. Zheng (2012), LFP')

ax.plot(Cc, mcap, marker = 'o', linestyle = '-', color = Cols[2], label = 'Current Work, NMC')
ax.plot(Cc_HZ_NMC, mcap_HZ_NMC, marker = 's', linestyle = '', color = '#0db39e', label = 'H. Zheng (2012), NMC')
ax.plot(Cc_HZ_LFP, mcap_HZ_LFP, marker = 'd', linestyle = '--', color = '#748cab', label = 'H. Zheng (2012), LFP')
ax.plot(Cc_DYW, mcap_DYW, marker = 'v', linestyle = ':', color = '#8a5a44', label = 'D.Y.W. Yu (2006), LFP')

ax.set_ylabel('Maximum Capacity [mAh/cm$^2$]', fontsize = 15)
ax.set_xlabel('C-rate [h$^{-1}$]', fontsize = 15)

plt.legend(loc='upper right')

#%%

fig, ax= plt.subplots(figsize=(9,5))



#dd=D[D['Source']== 'D.Y.W. Yu (2006)']

#Z=dfall.pivot_table(index='C-rate(rnd)', columns='Thickness(um)', values = 'Avg_DCapacity(Ah/cm2)').values
#t_unique = np.sort(dfall['Thickness(um)'].unique())
#C_unique = np.sort(dfall['C-rate(rnd)'].unique())

plt.contourf(t_unique, C_unique, Z)

#for i, z in enumerate(Z):
#    plt.plot(t_unique, z, label='C = {}'.format(C_unique[i]), marker = 'o')

#ax.set_ylabel('Q/A [mAh/cm$^2$]', fontsize = 15)
#ax.set_xlabel('Thickness [$\mu m$]', fontsize = 15)ms = cr/2

#plt.legend(loc='upper left', ncol=4)


#%%

#This is the Critical Thickness vs C-rate

fig, ax= plt.subplots(figsize=(13,6))

ax.set_ylim((15, 220))
ax.set_xlim((9e-2, 21))

ms = 10
lw = 2

ax.errorbar(Cc, tc, yerr = tce, marker = 'o', linestyle = '-', color = Cols[2], capsize = 5, markersize=ms, linewidth = lw, label = 'Current Work, NMC')

ax.errorbar(Cc_2[1:], tc_2[1:], yerr = tce_2[:,1:], lolims=True, marker = '', linestyle = 'none', color = Cols[2], capsize = 5, markersize=ms, linewidth = lw)
ax.errorbar(Cc_2[1:], tc_2[1:], yerr = tce_2[:,1:], marker = 'o', linestyle = 'none', color = Cols[2], capsize = 5, markersize=ms, linewidth = lw)

tce_4 = np.array([[tce_2[0,0]], [tce_2[1,0]+10]])
ax.errorbar(Cc_2[0], tc_2[0], yerr = tce_4, lolims=True, marker = '', linestyle = 'none', color = Cols[2], capsize = 5, markersize=ms, linewidth = lw)
ax.errorbar(Cc_2[0], tc_2[0], yerr = tce_4, marker = 'o', linestyle = 'none', color = Cols[2], capsize = 5, markersize=ms, linewidth = lw)

ax.errorbar(Cc_3, tc_3, yerr = tce_3, uplims=True, marker = '', linestyle = 'none', color = Cols[2], capsize = 5, markersize=ms, linewidth = lw)
ax.errorbar(Cc_3, tc_3, yerr = tce_3, marker = 'o', linestyle = 'none', color = Cols[2], capsize = 5, markersize=ms, linewidth = lw)

#ax.errorbar(Cc, tc, yerr = tce, marker = 'o', linestyle = '-', color = Cols[2], capsize = 5, label = 'Current Work, NMC')
ax.errorbar(Cc_HZ_NMC, tc_HZ_NMC, yerr = tce_HZ_NMC, marker = 's', linestyle = '', color = '#0db39e', capsize = 5, markersize=ms, linewidth = lw, label = 'H. Zheng (2012), NMC')
ax.errorbar(Cc_HZ_LFP, tc_HZ_LFP, yerr = tce_HZ_LFP, marker = 'd', linestyle = '--', color = '#748cab', capsize = 5, markersize=ms, linewidth = lw, label = 'H. Zheng (2012), LFP')
ax.errorbar(Cc_DYW, tc_DYW, yerr = tce_DYW, marker = 'v', linestyle = ':', color = '#8a5a44', capsize = 5, markersize=ms, linewidth = lw, label = 'D.Y.W. Yu (2006), LFP')


plt.xscale('log')


ax.set_ylabel('Critical Thickness [$\mu m$]', fontsize = 15)
ax.set_xlabel('C-rate [h$^{-1}$]', fontsize = 15)

plt.legend(loc='upper right')


