#%%
import numpy as np
import Cycler
import main
import pandas as pd
import matplotlib.pyplot as plt

#%%

OPdf = Cycler.get_overpotential('220203_NMC','03')


#%%

x=np.array([i for i in range(10)])

print(x)

index = np.array([False] * len(x))
index[3] = True

index2 = np.array([False] * len(x))
index2[1] = True

index3 = index | index2

x[index3] = -1

print(x)

#%%

df_Charge_Discharge = pd.DataFrame(columns=['Cycle_Index', 'Step_Index','Charge_Capacity(mAh/cm2)','Charge_Capacity(mAh/gAM)', 'Discharge_Capacity(mAh/cm2)','Discharge_Capacity(mAh/gAM)', 'Voltage(V)', 'C-rate(1/h)'])


DataDirectory = "Data/"
BatchLabel = '211202_NMC'
SampleID='02'

df = Cycler.get_PointData(BatchLabel, SampleID, DataDirectory, Properties=['Cycle_Index', 'Step_Index', 'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)', 'Voltage(V)', 'Current(A)'])
  
df_Charge_Discharge.loc[:,'Cycle_Index'] = df.loc[:,'Cycle_Index']
    
   
#%%



dfall, dfCrit = main.dfall_dfCrit(read_json=False)

#%%


SampleList = {
                        #'220203_NMC': ['03', '04', '06', '18', '21', '22', '23'],
                        #'220203_NMC': ['03', '04', '05', '06', '09', '17', '18', '21', '22', '23'],
                        '220621_LFP': ['01', '02', '03', '04', '06', '07', '08', '11', '12', '13', '15', '16', '19', '20', '22', '23'],
                    }
        
        #Not: '05', '09', '10', '14', '17', '18', '21' (21 data is coming)
#collecting batch cycle data 
df = Cycler.get_BatchCycles(SampleList)

#%%

fig, ax= plt.subplots(figsize=(13,6))

Cols = ['#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6', '#ee9b00', '#ca6702', '#bb3e03', '#ae2012', '#9b2226']

def colfun(i):
    return Cols[np.mod(i-1,len(Cols))]


for i, (Sample, group) in enumerate(df.loc[df['Cycle']<12,:].groupby('Sample')):
    
    thickness  =  group['Thickness(um)'].unique()[0]
    group.plot(x='Discharge_Current(mA/cm2)', y='Discharge_Capacity(mAh/gAM)', 
               color = colfun(i), 
               label = '{:.0f} $\mu m$'.format(thickness), 
               linestyle = '-', 
               marker = 'o', 
               markersize = (thickness-74)/(158-74)*6 + 2,
               ax=ax)
    
ax.set_xscale('log')

ax.set_ylabel('Discharge_Capacity(mAh/gAM)')
#%%
fig, ax= plt.subplots(figsize=(13,6))

Cols = ['#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6', '#ee9b00', '#ca6702', '#bb3e03', '#ae2012', '#9b2226']

def colfun(i):
    return Cols[np.mod(i-1,len(Cols))]



for i, (Sample, group) in enumerate(df.loc[df['Cycle']<12,:].groupby('Cycle')):
    
    group = group.sort_values(by='Thickness(um)')
    
    current  =  group['Discharge_Current(mA/cm2)'].mean()
    
    group.plot(x='Thickness(um)', y='Discharge_Capacity(mAh/gAM)', 
               color = colfun(i), 
               label = '{:.3f}'.format(current), 
               linestyle = '-', 
               marker = 'o', 
               markersize = 6,
               ax=ax)
    

#%%
fig, ax= plt.subplots(figsize=(13,6))

Cols = ['#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6', '#ee9b00', '#ca6702', '#bb3e03', '#ae2012', '#9b2226']

def colfun(i):
    return Cols[np.mod(i-1,len(Cols))]

Sample = '14'

Index = (df['Cycle']<30) & (df['Sample']==Sample)

df.loc[Index].plot(x='Discharge_Current(mA/cm2)', y='Discharge_Capacity(mAh/gAM)', 
               color = colfun(i), 
               label = str(Sample), 
               linestyle = '-', 
               marker = 'o', 
               markersize = (thickness-74)/(158-74)*6 + 2,
               ax=ax)
    
ax.set_xscale('linear')


#%%

Batch = '220203_NMC'
Sample = '23'

df = Cycler.get_PointData(Batch, Sample)



fig, ax= plt.subplots(figsize=(13,6))

Cols = ['#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6', '#ee9b00', '#ca6702', '#bb3e03', '#ae2012', '#9b2226']

def colfun(i):
    return Cols[np.mod(i-1,len(Cols))]


for i, ((Cyc, Stp), group) in enumerate(df.groupby(['Cycle_Index', 'Step_Index'])):
    
    group.plot(x='Test_Time(s)', y='Current(A)', 
               color = colfun(int(Stp)), 
               linestyle = '-', 
               marker = '.',
               label = "C : {}, S : {}".format(Cyc,Stp),
               ax=ax)
    


#%%


Batch = '220203_NMC'
Sample = '23'

df_Cyc = Cycler.get_CycleData(Batch, Sample)
df_Stp = Cycler.get_StepData(Batch, Sample)
df_Pnt = Cycler.get_PointData(Batch, Sample)


fig, ax= plt.subplots(figsize=(13,6))

ax2 = ax.twinx()

for i, ((Cyc, Stp), group) in enumerate(df_Pnt.groupby(['Cycle_Index', 'Step_Index'])):
    
    group.plot(x='Test_Time(s)', y='Current(A)', 
               color = colfun(int(Stp)), 
               linestyle = '-', 
               marker = '.',
               markersize = 1,
               ax=ax)
    
    ind = (df_Stp['Step_ID'] == Stp) &  (df_Stp['Cyc_ID'] == Stp)
    df_Stp


ax.get_legend().remove()

#%%

df_Cyc.plot(x='Cycle_ID', y='Discharge_Current(mA)', 
               color = 'k', 
               linestyle = '-', 
               marker = '.',
               ax=ax)

df_Cyc.plot(x='Cycle_ID', y='Discharge_Capacity(mAh)', 
               color = 'r', 
               linestyle = '-', 
               marker = '.',
               ax=ax2)






for i, ((Cyc, Stp), group) in enumerate(df.groupby(['Cycle_Index', 'Step_Index'])):
    
    group.plot(x='Test_Time(s)', y='Current(A)', 
               color = colfun(int(Stp)), 
               linestyle = '-', 
               marker = '.',
               label = "C : {}, S : {}".format(Cyc,Stp),
               ax=ax)
    
    
    
    

