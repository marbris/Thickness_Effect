

#%%

import Cycler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import main

#%%


#dfall, dfCrit = main.dfall_dfCrit(read_json = False, write_json = True)
dfall, dfCrit = main.dfall_dfCrit()

#%%


fig, ax= plt.subplots(figsize=(13,6))

index = (~dfall['Overpotential(V)'].isnull()) & (dfall['Cycle']<30) & (dfall['Batch'] == '211202_NMC')

markers = ['o','v']
def markfun(batch):
    if batch == '211202_NMC': 
        return markers[0]
    elif batch == '220203_NMC': 
        return markers[1]
    else:
        return '.'
    
    return markers[np.mod(i,len(markers))]


for i, ((batch, sample), group) in enumerate(dfall.loc[index,:].groupby(by = ['Batch', 'Sample'])):
    
    group.plot(x='Avg_Current(mA/cm2)', y = 'Avg_Overpotential(V)', yerr = 'Std_Overpotential(V)', marker=markfun(batch), ax=ax, label = 'B: {}, S: {}'.format(batch,sample))
    

ax.set_xscale('log')   
    
ax.set_ylabel('Overpotential (V)')

handles, labels = ax.get_legend_handles_labels()

ax.legend(handles, labels, 
          loc = 'upper left', 
          ncol = 2)
# %%
