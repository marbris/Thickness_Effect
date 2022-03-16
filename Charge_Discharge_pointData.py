#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 18:05:52 2022

@author: martin
"""


import Cycler
from matplotlib import pyplot as plt
import numpy as np


SMALL_SIZE = 15
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



#%% Charge discharge curves for all batteries. 

fig, ax= plt.subplots(figsize=(13,6))

DataDirectory = "Data/"
BatchLabel = '211202_NMC'

SL = Cycler.get_SampleList(DataDirectory)

diameter = 1.23 #cm
area = (diameter/2)**2*3.14 #electrode area, cm2
tt=[]
cc=[]

#PlotList = SL[BatchLabel]
PlotList = ['02', '13', '19']

for SampleID in PlotList:
    
    
    
    SampleInfo, BatchInfo = Cycler.get_SampleInfo(BatchLabel, SampleID, DataDirectory)
    Prog = SampleInfo['Cycler_Program']
    
    Thickness = SampleInfo['ECCthickness'] - BatchInfo['CurrentCollector']['Thickness'] #micrometer
    tt.append(Thickness)
    Mass = (SampleInfo['ECCmass']-BatchInfo['CurrentCollector']['Mass'])*1e-3 #grams
    
    Slurry_Mass = BatchInfo['Slurry']['AM']['Mass'] + BatchInfo['Slurry']['Binder']['Mass']*BatchInfo['Slurry']['Binder']['Binder_Concentration'] + BatchInfo['Slurry']['Carbon']['Mass']
    AM_Mass_frac = BatchInfo['Slurry']['AM']['Mass']/Slurry_Mass
    AM_Mass = Mass * AM_Mass_frac #grams
    
    df = Cycler.get_PointData(BatchLabel, SampleID, DataDirectory, Properties=['Cycle_Index', 'Step_Index', 'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)', 'Voltage(V)', 'Current(A)'])
  
    df.loc[:,'Current(mA/cm2)'] = df.loc[:,'Current(A)']*1e3/area
    df.loc[:,'Discharge_Capacity(mAh/cm2)'] = df.loc[:,'Discharge_Capacity(Ah)']*1e3/area
    df.loc[:,'Charge_Capacity(mAh/cm2)'] = df.loc[:,'Charge_Capacity(Ah)']*1e3/area
    
    df.loc[:,'Discharge_Capacity(mAh/gAM)'] = df.loc[:,'Discharge_Capacity(Ah)']*1e3/AM_Mass
    df.loc[:,'Charge_Capacity(mAh/gAM)'] = df.loc[:,'Charge_Capacity(Ah)']*1e3/AM_Mass
    
    df.loc[:,'C-rate(1/h)'] = df.loc[:,'Current(mA/cm2)']/df.loc[df['Cycle_Index']==2,'Discharge_Capacity(mAh/cm2)'].max()
    
    
    #lw = (Thickness-46)/(194-46)*3+0.5
    lw= 2
    #Xs = 'Charge_Capacity(mAh/cm2)'
    Xs = 'Charge_Capacity(mAh/gAM)'
    df[(df['Cycle_Index']==2) & (df['Step_Index']==2)].plot(y= 'Voltage(V)' , x=Xs, ax=ax, linewidth = lw, color = '#22223b')
    df[(df['Cycle_Index']==31) & (df['Step_Index']==52)].plot(y= 'Voltage(V)' , x=Xs, ax=ax, linewidth = lw, color = '#9a8c98')
   
    #Xs = 'Discharge_Capacity(mAh/cm2)'
    Xs = 'Discharge_Capacity(mAh/gAM)'
    df[(df['Cycle_Index']==2) & (df['Step_Index']==4)].plot(y= 'Voltage(V)' , x=Xs, ax=ax, linewidth = lw, color = '#a4133c')
    df[(df['Cycle_Index']==31) & (df['Step_Index']==54)].plot(y= 'Voltage(V)' , x=Xs, ax=ax, linewidth = lw, color = '#ff4d6d')
    
    xpos = df.loc[(df['Cycle_Index']==2) & (df['Step_Index']==4),Xs].max()
    cc.append(xpos)
    #ypos = 2.78 - np.mod(len(tt)-1,3)*0.07
    
    #if Thickness == 46: ypos -= 0.07*3
    
    #ax.text(xpos, ypos, str(Thickness), color = 'k', fontsize = 14, horizontalalignment = 'center', verticalalignment = 'top')
    
    
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[-8:-4], ['Charge Cycle #2', 'Charge Cycle #31', 'Discharge Cycle #2', 'Discharge Cycle #31'], fontsize = 14, bbox_to_anchor=(1,0.72), loc='center right', framealpha = 0)

#plt.xlabel('Areal Capacity [mAh/cm$^2$]', fontsize = 18)
plt.xlabel('Specific Capacity [mAh/g$_{AM}$]', fontsize = 18)
plt.ylabel('Voltage [V]', fontsize = 18)

plt.ylim((2.48,4.45))
#plt.xlim((-0.01,8.5))
plt.xlim((-5,250))

ax.set_yticks([2.8, 3.6, 4.4])

tt2, cc2 = zip(*sorted(zip(tt, cc), key=lambda k: k[1], reverse=True))

for i, t in enumerate(tt2):
    ypos = 2.78 - np.mod(i,3)*0.09
    #ypos = 2.75
    xpos = cc2[i]
    fs = 18
    ax.text(xpos, ypos, str(t), color = 'k', fontsize = fs, horizontalalignment = 'center', verticalalignment = 'top')

#xpos = 0
xpos = 150
ax.text(xpos, 2.75, 'Cathode \nThickness [$\mu$m] :', color = 'k', fontsize = 14, horizontalalignment = 'left', verticalalignment = 'top')

#xpos=1
xpos=100
ax.text(xpos, 3.1, 'C/10', color = 'k', fontsize = 30, horizontalalignment = 'center')

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    