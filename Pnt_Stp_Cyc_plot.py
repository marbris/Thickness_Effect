
#%%

import numpy as np
import Cycler
import main
import pandas as pd
import matplotlib.pyplot as plt


#%%

Batch = '220621_LFP'
Sample = '01'

df_Cyc = Cycler.get_CycleData(Batch, Sample)
df_Stp = Cycler.get_StepData(Batch, Sample)
df_Pnt = Cycler.get_PointData(Batch, Sample)


fig, ax= plt.subplots(figsize=(16,6))

#ax.set_ylim((-0.01, 0.003))


Cols = ['#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6', '#ee9b00', '#ca6702', '#bb3e03', '#ae2012', '#9b2226']
def colfun(i):
    return Cols[np.mod(i-1,len(Cols))]


#ax2 = ax.twinx()

for i, ((Cyc, Stp), group) in enumerate(df_Pnt.groupby(['Cycle', 'Step'])):
    
    group.plot(x='Test_Time(s)', y='Voltage(V)',#'Current(mA)', 
               color = colfun(int(Cyc)), 
               linestyle = '-', 
               marker = '.',
               markersize = 2,
               ax=ax)
    
    Step_start_time = group['Test_Time(s)'].min()
    Step_end_time = group['Test_Time(s)'].max()
    
    
    
    ind = (df_Stp['Step'] == Stp) &  (df_Stp['Cycle'] == Cyc)
    print('({},{})'.format(Cyc,Stp))
    
    if ind.sum() > 0:
        
        
        Y = df_Stp.loc[ind,'Current_High(mA)'].values[0]
        if group['Current(mA)'].sum()<0:
            Y=-Y
            
        plt.hlines(Y,Step_start_time,Step_end_time,
                    color = 'k', 
                    linestyle = '-', 
                    linewidth = 6, 
                    alpha = 0.5)
        
        #X = np.mean([Step_start_time,Step_end_time])
        #Y = 0.0015 + np.mod(i,5)*0.0005
        #plt.text(X,Y,'({},{})'.format(Cyc,Stp), horizontalalignment = 'center')
        
        #Y = -df_Stp.loc[ind,'Current_Low(mA)'].values[0]*1e-3
        #plt.hlines(Y,Step_start_time,Step_end_time,
        #            color = colfun(int(Stp)), 
        #            linestyle = '-', 
        #            linewidth = 4, 
        #            alpha = 0.5)
    
for Cyc in df_Cyc['Cycle'].tolist():
    
    Y = -df_Cyc.loc[df_Cyc['Cycle']==Cyc,'Discharge_Current(mA)'].values[0]
    x0 = df_Pnt.loc[df_Pnt['Cycle']==Cyc,'Test_Time(s)'].min()
    x1 = x0 + df_Cyc.loc[df_Cyc['Cycle']==Cyc,'Cycle_Duration(s)'].values[0]
    plt.hlines(Y,x0,x1,
                color = 'r', 
                linestyle = '-', 
                linewidth = 8, 
                alpha = 0.5)

    Y = df_Cyc.loc[df_Cyc['Cycle']==Cyc,'Charge_Current(mA)'].values[0]
    plt.hlines(Y,x0,x1,
                color = 'b', 
                linestyle = '-', 
                linewidth = 8, 
                alpha = 0.5)

ax.get_legend().remove()
plt.show()


