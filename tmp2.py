
#%%

import numpy as np
import Cycler
import main
import pandas as pd
import matplotlib.pyplot as plt


#%%

Batch = '220203_NMC'
Sample = '23'

df_Cyc = Cycler.get_CycleData(Batch, Sample)
df_Stp = Cycler.get_StepData(Batch, Sample)
df_Pnt = Cycler.get_PointData(Batch, Sample)


fig, ax= plt.subplots(figsize=(13,6))


Cols = ['#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6', '#ee9b00', '#ca6702', '#bb3e03', '#ae2012', '#9b2226']
def colfun(i):
    return Cols[np.mod(i-1,len(Cols))]


#ax2 = ax.twinx()

for i, ((Cyc, Stp), group) in enumerate(df_Pnt.groupby(['Cycle_Index', 'Step_Index'])):
    
    group.plot(x='Test_Time(s)', y='Current(A)', 
               color = colfun(int(Stp)), 
               linestyle = '-', 
               marker = '.',
               markersize = 2,
               ax=ax)
    
    Step_start_time = group['Test_Time(s)'].min()
    Step_end_time = group['Test_Time(s)'].max()
    
    ind = (df_Stp['Step_ID'] == Stp) &  (df_Stp['Cycle_ID'] == Cyc)
    
    if ind.sum() > 0:
        
        
        Y = df_Stp.loc[ind,'Current_High(mA)'].values[0]*1e-3
        if group['Current(A)'].sum()<0:
            Y=-Y
            
        plt.hlines(Y,Step_start_time,Step_end_time,
                    color = 'k', 
                    linestyle = '-', 
                    linewidth = 6, 
                    alpha = 0.5)
        
        #Y = -df_Stp.loc[ind,'Current_Low(mA)'].values[0]*1e-3
        #plt.hlines(Y,Step_start_time,Step_end_time,
        #            color = colfun(int(Stp)), 
        #            linestyle = '-', 
        #            linewidth = 4, 
        #            alpha = 0.5)
    

ax.get_legend().remove()