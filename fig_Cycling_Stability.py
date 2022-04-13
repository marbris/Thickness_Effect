

#%%

import Cycler
import pandas as pd
import matplotlib.pyplot as plt


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


#%%


SampleList = { '220209_NMC': ['02', '11', '33', '37']}


df = Cycler.get_BatchCycles(SampleList)

#%%

fig, ax = plt.subplots(figsize=(Col2,Col1*AR))

ax2 = ax.twinx()

xmin = -0.5
xmax = 50.5
ax.set_xlim((xmin, xmax))

ax.set_ylabel('Areal Capacity \n[mAh/cm$^2$]')
ax2.set_ylabel('Coulombic \nEfficicency [%]')
ax.set_xlabel('Cycle')

index = (df['Cycle'] < df['Cycle'].max()-1) & (df['Sample']=='37')

df.loc[index,:].plot(x='Cycle', y = 'Coulombic_Efficiency(%)', 
                     ax=ax2, 
                     marker = 'o', 
                     label = 'Coulombic Efficiency',
                     color = 'k',
                     zorder = 1)
plt.hlines(100,xmin,xmax,color = 'k',linestyle=':',linewidth=1,zorder=0)

df.loc[index,:].plot(x='Cycle', y = 'Discharge_Capacity(mAh/cm2)', 
                     ax=ax, 
                     marker = 'v', 
                     label = 'Discharge Capacity',
                     color = 'r', 
                     alpha = 0.6,
                     zorder = 3)
df.loc[index,:].plot(x='Cycle', y = 'Charge_Capacity(mAh/cm2)', 
                     ax=ax, 
                     marker = '^', 
                     label = 'Charge Capacity',
                     color = 'b', 
                     alpha = 0.6, 
                     zorder = 2)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, 
          framealpha = 0, 
          fontsize = 8,
          bbox_to_anchor=(1.00,0.04), 
          loc = 'lower right',
          columnspacing=0.5, 
          handletextpad = 0.3, 
          labelspacing = 0.3)


handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles, labels, 
          framealpha = 0.0, 
          fontsize = 8,
          bbox_to_anchor=(1.00,0.96), 
          loc = 'upper right',
          columnspacing=0.5, 
          handletextpad = 0.3, 
          labelspacing = 0.3)

#for sample, group in df.groupby(by = 'Sample'):
#    index = group['Cycle'] < group['Cycle'].max()
#    
#    group.loc[index,:].plot(x='Cycle', y = 'Coulombic_Efficiency', ax=ax, marker = 'o', label = sample)
    
#    group.loc[index,:].plot(x='Cycle', y = 'Discharge_Capacity(mAh/cm2)', ax=ax2, marker = 'o', label = sample)
#    group.loc[index,:].plot(x='Cycle', y = 'Charge_Capacity(mAh/cm2)', ax=ax2, marker = 'o', label = sample)
    
    
fig.tight_layout()
plt.show()
