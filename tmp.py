#%%
import numpy as np
import Cycler
import pandas as pd

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
    
   


