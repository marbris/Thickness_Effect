
#%%
import Cycler

import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
import main

SMALL_SIZE = 10
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

dfall, dfCapCrit, dfOPCrit = main.dfall_dfCrit()


#%%

index = (dfall['Batch']=='211202_NMC') & ~np.isnan(dfall['Avg_Current(mA/cm2)'])

Y = dfall.loc[index, 'Avg_Current(mA/cm2)'].to_numpy(dtype = float)
Z = dfall.loc[index, 'Avg_DCapacity(mAh/cm2)'].to_numpy(dtype = float)
X = dfall.loc[index, 'Thickness(um)'].to_numpy(dtype = float)

x_2d, y_2d = np.meshgrid(X,Y)

z_2d = np.empty(x_2d.shape)

Yunique = np.unique(Y)

for xi in range(len(X)):
    indexx = (dfall['Thickness(um)'] == X[xi])
    for yi in range(len(Y)):
        indexy = (dfall['Avg_Current(mA/cm2)'] == Y[yi])
        
        print(f'{xi}, {yi}')
        
        index3 = indexy & indexx & index
        
        z_2d[xi,yi] = dfall.loc[index3, 'Avg_DCapacity(mAh/cm2)'].mean()


# %%
