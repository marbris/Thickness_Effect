#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 14:57:55 2022

@author: martin
"""
#%%
import Cycler
from matplotlib import pyplot as plt
import numpy as np



SMALL_SIZE = 14
MEDIUM_SIZE = 15
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


#%% AreaCap vs Cycle


fig, ax= plt.subplots(figsize=(13,6))


    
Crates = [ [0.1, 0.5, 0.8, 1.2, 1.8, 2.7, 4.2, 6.4, 9.8, 15.1],
           [0.1, 0.6, 0.9, 1.3, 2.1, 3.2, 4.8, 7.4, 11.3, 17.4],
           [0.1, 0.7, 1, 1.6, 2.4, 3.6, 5.6, 8.5, 13.1, 20] ]


Cols = ['black', '#219ebc', '#e36414', '#606c38', 'green']

prog_cols = {"Martin_cycles_1": Cols[1],
             "Martin_cycles_2": Cols[2],
             "Martin_cycles_3": Cols[3]}

DataDirectory = "Data/"
BatchLabel = '211202_NMC'

SL = Cycler.get_SampleList(DataDirectory)

diameter = 1.23 #cm
area = (diameter/2)**2*3.14 #electrode area, cm2
tt=[]
pp=[]

for SampleID in SL[BatchLabel]:
    
    SampleInfo, BatchInfo = Cycler.get_SampleInfo(BatchLabel, SampleID, DataDirectory)
    Prog = SampleInfo['Cycler_Program']
    
    Thickness = SampleInfo['ECCthickness'] - BatchInfo['CurrentCollector']['Thickness'] #micrometer
    tt.append(Thickness)
    Mass = (SampleInfo['ECCmass']-BatchInfo['CurrentCollector']['Mass'])*1e-3 #grams
    
    Slurry_Mass = BatchInfo['Slurry']['AM']['Mass'] + BatchInfo['Slurry']['Binder']['Mass']*BatchInfo['Slurry']['Binder']['Binder_Concentration'] + BatchInfo['Slurry']['Carbon']['Mass']
    AM_Mass_frac = BatchInfo['Slurry']['AM']['Mass']/Slurry_Mass
    AM_Mass = Mass * AM_Mass_frac
    
    
    
    df = Cycler.get_CycleData(BatchLabel, SampleID, DataDirectory, Properties=['Cycle_ID', 'Discharge_Capacity(mAh)', 'Discharge_Current(mA)'])
    df.loc[:,'Current(mA/cm2)'] = -df.loc[:,'Discharge_Current(mA)']/area
    df.loc[:,'Capacity(mAh/cm2)'] = df.loc[:,'Discharge_Capacity(mAh)']/area
    df.loc[:,'Capacity(mAh/gAM)'] = df.loc[:,'Discharge_Capacity(mAh)']/AM_Mass
    df.loc[:,'Cycler_Program'] = int(SampleInfo['Cycler_Program'][-1:])
    pp.append(int(SampleInfo['Cycler_Program'][-1:]))
    df.loc[:,'C-rate(1/h)'] = df.loc[:,'Current(mA/cm2)']/df.loc[df['Cycle_ID']==2,'Capacity(mAh/cm2)'].values[0]
    
    #for i in range(3):
    #    for cr in Crates[i]:
    #        df.loc[(abs(df.loc[:,'C-rate(1/h)']/cr-1)<=0.18) & (df.loc[:,'Cycler_Program'] == i+1),'C-rate(rnd)'] = cr
    #        df.loc[abs(df.loc[:,'C-rate(1/h)']/cr-1)<=0.18,'C-rate(rnd-frac)'] = df.loc[:,'C-rate(1/h)']/cr-1
        
    
    df.iloc[:-1].plot(x = 'Cycle_ID', y='Capacity(mAh/cm2)', label = '{} $\mu$m'.format(Thickness), ax=ax, marker = 'o', color = prog_cols[Prog], markersize = (Thickness-46)/(194-46)*10+2)
    
    


#this is a dummy line just to align the legend.
ax.plot(np.zeros(1), np.zeros([1,2]), color='w', alpha=0, label=' ')
pp.append(3)
tt.append(0)

#this is a dummy line just to align the legend.
for i in range(3):
    ax.plot(np.zeros(1), np.zeros([1,2]), color='w', alpha=0, label=' ')
    pp.append(1)
    tt.append(0)


    
Cycle_ID_Crate = [1.5, 4, 7, 10, 13, 16, 19, 22, 25, 28, 30.5]

N = len(Cycle_ID_Crate)-1
for i in range(1,N):
    
    y = (11 - (i - 1)/(N-2)*9)/area
    ax.text(Cycle_ID_Crate[i], y, str(Crates[0][i]), color = prog_cols["Martin_cycles_1"], fontsize = 18, horizontalalignment = 'center')
    ax.text(Cycle_ID_Crate[i], y-0.6, str(Crates[1][i]), color = prog_cols["Martin_cycles_2"], fontsize = 18, horizontalalignment = 'center')
    ax.text(Cycle_ID_Crate[i], y-1.2, str(Crates[2][i]), color = prog_cols["Martin_cycles_3"], fontsize = 18, horizontalalignment = 'center')

y = (11 - (0 - 1)/(N-2)*9)/area
ax.text(Cycle_ID_Crate[0], y-0.6, str(Crates[0][0]), color = 'k', fontsize = 18, horizontalalignment = 'center')
ax.text(Cycle_ID_Crate[-1], y-0.6, str(Crates[0][0]), color = 'k', fontsize = 18, horizontalalignment = 'center')

ax.text(7, 9, 'C-rates', color = 'k', fontsize = 18, horizontalalignment = 'center', rotation = -25)

plt.ylim((0,11))
plt.xlim((0,33))

handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
labels, handles, tt2, pp2 = zip(*sorted(zip(labels, handles, tt, pp), key=lambda k: (np.mod(k[3]+1,3), k[2]), reverse=True))
ax.legend(handles, labels, ncol = 3, loc='upper center', bbox_to_anchor=(0.6,1), framealpha = 0)


plt.xlabel('Cycle', fontsize = 18)
plt.ylabel('Discharge Capacity [mAh/cm$^2$]', fontsize = 18)


#%% MassCap vs Cycle


fig, ax= plt.subplots(figsize=(13,6))


    
Crates = [ [0.1, 0.5, 0.8, 1.2, 1.8, 2.7, 4.2, 6.4, 9.8, 15.1],
           [0.1, 0.6, 0.9, 1.3, 2.1, 3.2, 4.8, 7.4, 11.3, 17.4],
           [0.1, 0.7, 1, 1.6, 2.4, 3.6, 5.6, 8.5, 13.1, 20] ]


Cols = ['black', '#219ebc', '#e36414', '#606c38', 'green']

prog_cols = {"Martin_cycles_1": Cols[1],
             "Martin_cycles_2": Cols[2],
             "Martin_cycles_3": Cols[3]}

DataDirectory = "Data/"
BatchLabel = '211202_NMC'

SL = Cycler.get_SampleList(DataDirectory)

diameter = 1.23 #cm
area = (diameter/2)**2*3.14 #electrode area, cm2
tt=[]
pp=[]

for SampleID in SL[BatchLabel]:
    
    SampleInfo, BatchInfo = Cycler.get_SampleInfo(BatchLabel, SampleID, DataDirectory)
    Prog = SampleInfo['Cycler_Program']
    
    Thickness = SampleInfo['ECCthickness'] - BatchInfo['CurrentCollector']['Thickness'] #micrometer
    tt.append(Thickness)
    Mass = (SampleInfo['ECCmass']-BatchInfo['CurrentCollector']['Mass'])*1e-3 #grams
    
    Slurry_Mass = BatchInfo['Slurry']['AM']['Mass'] + BatchInfo['Slurry']['Binder']['Mass']*BatchInfo['Slurry']['Binder']['Binder_Concentration'] + BatchInfo['Slurry']['Carbon']['Mass']
    AM_Mass_frac = BatchInfo['Slurry']['AM']['Mass']/Slurry_Mass
    AM_Mass = Mass * AM_Mass_frac
    
    
    
    df = Cycler.get_CycleData(BatchLabel, SampleID, DataDirectory, Properties=['Cycle_ID', 'Discharge_Capacity(mAh)', 'Discharge_Current(mA)'])
    df.loc[:,'Current(mA/cm2)'] = -df.loc[:,'Discharge_Current(mA)']/area
    df.loc[:,'Capacity(mAh/cm2)'] = df.loc[:,'Discharge_Capacity(mAh)']/area
    df.loc[:,'Capacity(mAh/gAM)'] = df.loc[:,'Discharge_Capacity(mAh)']/AM_Mass
    df.loc[:,'Cycler_Program'] = int(SampleInfo['Cycler_Program'][-1:])
    pp.append(int(SampleInfo['Cycler_Program'][-1:]))
    df.loc[:,'C-rate(1/h)'] = df.loc[:,'Current(mA/cm2)']/df.loc[df['Cycle_ID']==2,'Capacity(mAh/cm2)'].values[0]
    
    #for i in range(3):
    #    for cr in Crates[i]:
    #        df.loc[(abs(df.loc[:,'C-rate(1/h)']/cr-1)<=0.18) & (df.loc[:,'Cycler_Program'] == i+1),'C-rate(rnd)'] = cr
    #        df.loc[abs(df.loc[:,'C-rate(1/h)']/cr-1)<=0.18,'C-rate(rnd-frac)'] = df.loc[:,'C-rate(1/h)']/cr-1
        
    
    df.iloc[:-1].plot(x = 'Cycle_ID', y='Capacity(mAh/gAM)', label = '{} $\mu$m'.format(Thickness), ax=ax, marker = 'o', color = prog_cols[Prog], markersize = (Thickness-46)/(194-46)*10+2)
    
    


#this is a dummy line just to align the legend.
ax.plot(np.zeros(1), np.zeros([1,2]), color='w', alpha=0, label=' ')
pp.append(3)
tt.append(0)

#this is a dummy line just to align the legend.
for i in range(3):
    ax.plot(np.zeros(1), np.zeros([1,2]), color='w', alpha=0, label=' ')
    pp.append(1)
    tt.append(0)


    
Cycle_ID_Crate = [1.5, 4, 7, 10, 13, 16, 19, 22, 25, 28, 30.5]

N = len(Cycle_ID_Crate)-1
for i in range(1,N):
    
    y = -25
    ax.text(Cycle_ID_Crate[i], y, str(Crates[0][i]), color = prog_cols["Martin_cycles_1"], fontsize = 18, horizontalalignment = 'center')
    ax.text(Cycle_ID_Crate[i], y-20, str(Crates[1][i]), color = prog_cols["Martin_cycles_2"], fontsize = 18, horizontalalignment = 'center')
    ax.text(Cycle_ID_Crate[i], y-40, str(Crates[2][i]), color = prog_cols["Martin_cycles_3"], fontsize = 18, horizontalalignment = 'center')

y = -25 #(11 - (0 - 1)/(N-2)*9)/area
ax.text(Cycle_ID_Crate[0], y, str(Crates[0][0]), color = 'k', fontsize = 18, horizontalalignment = 'center')
ax.text(Cycle_ID_Crate[-1], y, str(Crates[0][0]), color = 'k', fontsize = 18, horizontalalignment = 'center')

ax.text(7, 9, 'C-rates', color = 'k', fontsize = 18, horizontalalignment = 'center')

plt.ylim((-70,270))
plt.xlim((0,33))

handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
labels, handles, tt2, pp2 = zip(*sorted(zip(labels, handles, tt, pp), key=lambda k: (np.mod(k[3]+1,3), k[2]), reverse=True))
ax.legend(handles, labels, ncol = 3, loc='upper center', bbox_to_anchor=(0.65,1), framealpha = 0)


plt.xlabel('Cycle', fontsize = 18)
plt.ylabel('Discharge Capacity [mAh/g$_{AM}$]', fontsize = 18)
