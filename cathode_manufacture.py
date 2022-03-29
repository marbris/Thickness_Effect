
#%%

import Cycler
import matplotlib.pyplot as plt
import numpy as np




SampleList = Cycler.get_SampleList()


Solid_ratio = np.array([])
AM_ratio = np.array([])
C_ratio = np.array([])
PVDF_ratio = np.array([])

solid_mass = np.array([])

Batches = np.array([])

SL = list(SampleList.keys())
SL.sort()

for Batch in SL:
    BI = Cycler.get_BatchInfo(Batch)
    AM_Mass = BI['Slurry']['AM']['Mass']
    C_Mass = BI['Slurry']['Carbon']['Mass']
    B_Mass = BI['Slurry']['Binder']['Mass']
    B_Conc = BI['Slurry']['Binder']['Binder_Concentration']
    PVDF_Mass = B_Mass * B_Conc
    NMP_Mass = B_Mass * (1 - B_Conc)
    Solid_Mass = AM_Mass + C_Mass + PVDF_Mass
    Total_Mass = AM_Mass + C_Mass + B_Mass
    
    Solid_ratio = np.append(Solid_ratio, Solid_Mass/Total_Mass)
    
    AM_ratio = np.append(AM_ratio, AM_Mass/Solid_Mass)
    C_ratio = np.append(C_ratio, C_Mass/Solid_Mass)
    PVDF_ratio = np.append(PVDF_ratio, PVDF_Mass/Solid_Mass)
    solid_mass = np.append(solid_mass, Solid_Mass)
    
    
    
    
    
    
    

#%%

fig, ax= plt.subplots(figsize=(6,6))

ax2 = ax.twinx()

xx = [i for i in range(len(SL))]
ax.plot(xx, Solid_ratio, marker = 'o', linestyle = '', label='solid ratio')
ax.plot(xx, PVDF_ratio, marker = 'o', linestyle = '', label='PVDF ratio')
ax.plot(xx, AM_ratio, marker = 'o', linestyle = '', label='AM ratio')

ax2.plot(xx, solid_mass, marker = 's', color = 'r', linestyle = '', label='solid mass')

ax.set_xticks(xx)
ax.set_xticklabels(SL)

plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor")

fig.tight_layout()

