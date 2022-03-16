#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 13:13:20 2022

@author: martin
"""
import Cycler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

plt.rc('figure', dpi = 100)
#%%

# SampleList = {
#                     #'220203_NMC': ['10', '11', '15', '16', '24', '25'],
#                     #'220203_NMC': ['03', '04', '05', '06', '09', '17', '18', '21', '22', '23'],
#                     '220203_NMC': ['03', '04', '06', '18', '21', '22', '23'],
#                     '211202_NMC': ['02', '03', '05', '06', '07', '08', '09', '12', '13', '15', '16', '17', '18', '19']
#                  }




#%%


def get_overpotential(Batch, Sample, Plot=False, Excl_Cyc = [1,2], Dcap_Lim = 0.3, SG_poly = 2, SG_window = 11, Charge_Vlim = (4.2, 4.35)):
    
    if Plot:
        fig, ax= plt.subplots(figsize=(13,6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
    df = Cycler.get_PointData(Batch, Sample)
    df_cyc = Cycler.get_CycleData(Batch, Sample)
    
    OPdf = pd.DataFrame(columns=['Cycle', 'Charge_peak(V)', 'Discharge_peak(V)', 'Overpotential(V)'])
    OPdf.loc[:, 'Cycle'] = df_cyc['Cycle_ID'].unique()

    
    groups = df.groupby(by=['Cycle_Index','Step_Index'])
    
    dcap=np.array([])
    
    for name, group in groups:
        
        TotCap = group.loc[:, 'Current(A)'].sum() 
        if TotCap > 0:
            mode = "Charge"
        elif TotCap < 0:
            mode = "Discharge"
        else:
            mode = "Rest"    
        
    
        Dcap = df_cyc.loc[df_cyc['Cycle_ID']==name[0],'Discharge_Capacity(mAh)'].values
        dcap = np.append(dcap,Dcap)
        
        
        if (mode != "Rest") & (Dcap/dcap[0]>Dcap_Lim) & (name[0] not in Excl_Cyc):
    
            V = group.loc[:,'Voltage(V)'].to_numpy()
            if mode == 'Charge': 
                Q = group.loc[:,'Charge_Capacity(Ah)'].to_numpy()
            elif mode == 'Discharge': 
                Q = group.loc[:,'Discharge_Capacity(Ah)'].to_numpy()
                
            dQdV = np.gradient(Q,V)
            
            index = abs(dQdV) > 1e-10
            dQdV = dQdV[index]
            V = V[index]
                
            if (mode=='Charge'):
                
                index = (V>Charge_Vlim[0]) & (V<Charge_Vlim[1])
                if sum(index) >= SG_window:
                
                    dQdV_smooth = savgol_filter(dQdV,SG_window,SG_poly)
                    
                    vtemp = V[index]
                    dqdvtemp = dQdV_smooth[index]
                    
                    peak_index = dqdvtemp == dqdvtemp.max()
                    peakV = vtemp[peak_index]
                    OPdf.loc[OPdf['Cycle']==name[0], 'Charge_peak(V)'] = peakV
                    
                    if Plot:
                        plt.plot(vtemp,dqdvtemp, marker='.', linestyle ='-', color = colors[np.mod(name[0],len(colors))])
                        plt.plot(peakV, dqdvtemp.max(), marker = 'o', markersize = 10, alpha = 0.5, color = 'k')
                        
            elif (mode=='Discharge'):
                
                threshold_index = 70
                
                if len(V) - 1 <threshold_index:
                    index = (V > 0)
                else:
                    index = (V > V[threshold_index])
                
                if sum(index) >= SG_window:
    
                    dQdV_smooth = savgol_filter(dQdV,SG_window,SG_poly)
    
                    vtemp = V[index]
                    dqdvtemp = dQdV_smooth[index]
                    peak_index = dqdvtemp == dqdvtemp.min()
                    peakV = vtemp[peak_index]
    
                    OPdf.loc[OPdf['Cycle']==name[0], 'Discharge_peak(V)'] = peakV
                    
                    if Plot:
                        plt.plot(vtemp,dqdvtemp, marker='.', linestyle ='-', color = colors[np.mod(name[0],len(colors))])
                        plt.plot(peakV, dqdvtemp.min(), marker = 'o', alpha = 0.7, color = 'k')
    
                
                
    OPdf.loc[:,'Overpotential(V)'] = (OPdf.loc[:,'Charge_peak(V)'] + OPdf.loc[:,'Discharge_peak(V)'])/2
    
    
    #ax.set_ylim((-0.05, 0.05))
    if Plot:
        plt.show()
        ax.set_ylabel('dQ/dV [Ah/V]', fontsize = 16)
        ax.set_xlabel('Voltage [V]', fontsize = 16)
    
    return OPdf #, dcap







#%%

SampleList = {
                    #'220203_NMC': ['10', '11', '15', '16', '24', '25'],
                    #'220203_NMC': ['03', '04', '05', '06', '09', '17', '18', '21', '22', '23'],
                    '220203_NMC': ['03', '04', '06', '18', '21', '22', '23'],
                    '211202_NMC': ['02', '03', '05', '06', '07', '08', '09', '12', '13', '15', '16', '17', '18', '19']
                 }

#OPdf = Cycler.get_overpotential('211202_NMC', '19', Plot=True, Excl_Cyc = [], Curve_Volt=0.05)
OPdf = Cycler.get_overpotential('220203_NMC', '18', Plot=True, Excl_Cyc = [], Curve_Volt=0.05)

#%%

#fig, ax= plt.subplots(figsize=(13,6))




#%%

fig, ax= plt.subplots(figsize=(13,6))

OPdf.plot(x='Cycle',y='Charge_peak(V)', ax=ax, marker = '.', color = 'k')
OPdf.plot(x='Cycle',y='Discharge_peak(V)', ax=ax, marker = '.', color = 'r')
OPdf.plot(x='Cycle',y='Overpotential(V)', ax=ax, marker = '.', color = 'b')


#%%

SampleList = {
                    #'220203_NMC': ['10', '11', '15', '16', '24', '25'],
                    #'220203_NMC': ['03', '04', '05', '06', '09', '17', '18', '21', '22', '23'],
                    '220203_NMC': ['03', '04', '06', '18', '21', '22', '23'],
                    '211202_NMC': ['02', '03', '05', '06', '07', '08', '09', '12', '13', '15', '16', '17', '18', '19']
                 }
for sample in SampleList['220203_NMC']:
    OPdf = Cycler.get_overpotential('220203_NMC', sample, Plot=1)
    


#%%




