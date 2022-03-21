
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
                            

dfall, dfCrit = main.dfall_dfCrit()

#%% plotting areacap vs thickness for my sampels

Batch = '211202_NMC'

dfplot = dfall.loc[dfall['Batch'] == Batch, 
                   ('C-rate(rnd)', 
                    'Avg_C-rate(1/h)', 
                    'Thickness(um)', 
                    'Avg_DCapacity(mAh/cm2)')].copy()


dfCritplot = dfCrit.loc[dfCrit['Batch'] == Batch, 
                        ('C-rate_mean(1/h)', 
                         'C-rate(rnd)', 
                         'Thickness_max(um)',
                         'Avg_DCapacity_max(mAh/cm2)')].copy()

fig, ax= plt.subplots(figsize=(Col1,Col1*AR*1.2))    

ax.set_ylim((-0.5, 10))
ax.set_xlim((0, 200))

crts2 = [0.1, 0.6, 1.0, 1.3,1.6, 2.1, 2.4, 3.6, 5.6, 6.4]

cr_groups = np.array([1.0, 3.6, 13.1])
Cols = ['#219ebc', '#e36414', '#606c38']
markers = ['^','o', 'v']

def colorfun(G):
    #the color is determined by how many lower cr-groups there are
    color = Cols[G]
    return color
    
def sizefun(G, cr):
    if G == 0:
        ms = 1 + 4*cr
    elif G==1:
        ms = 2*cr
    elif G==2:
        ms = (cr-4.5)*3    
    return ms

def markfun(G):
    return markers[G]

#Im using these to sort the legend
gg=np.array([])
cc=np.array([])

for cr in crts2:
    G = int(3 - sum(cr <= cr_groups))
    gg = np.append(gg,G)
    cc = np.append(cc, cr)
    
    df3 = dfplot.loc[dfplot['C-rate(rnd)'] == cr].copy()
    label = '{:.1f}'.format(df3['Avg_C-rate(1/h)'].mean())
    
    df3.sort_values(by = ['Thickness(um)'], ignore_index=True, inplace = True)
    df3.plot(x = 'Thickness(um)', y = 'Avg_DCapacity(mAh/cm2)', ax = ax, marker = markfun(G), markersize = sizefun(G, cr), color = colorfun(G), linewidth = 1, label = label)#'C: {}'.format(cr))
    
    if G==1:
        
        dfCritplot[dfCritplot['C-rate(rnd)']==cr].plot(x='Thickness_max(um)', y='Avg_DCapacity_max(mAh/cm2)', marker = markfun(G), markersize = sizefun(G, cr)*2+2, color = colorfun(G), alpha = 0.3, ax=ax, label='_nolegend_')
    

NCol = len(np.unique(gg))
maxrows = 5
for ggi in np.unique(gg):
    N_dummies = maxrows - len(gg[gg==ggi])
    for i in range(N_dummies):
            ax.plot(np.zeros(1), np.zeros([1,2]), color='w', alpha=0, label=' ')
            gg = np.append(gg, ggi)
            cc = np.append(cc, -1)


handles, labels = ax.get_legend_handles_labels()
# sort both labels and handles by labels
labels, handles, tt2, pp2 = zip(*sorted(zip(labels, handles, cc, gg), key=lambda k: (np.mod(k[3]+1,3), k[2]), reverse=True))

ax.legend(handles, labels, 
          ncol = NCol, 
          framealpha = 0, 
          bbox_to_anchor=(-0.03,0.95), 
          loc = 'upper left',
          columnspacing=0.5, 
          handletextpad = 0.3, 
          labelspacing = 0.3)

ax.text(5, 8.9, 'C-rates', color = 'k', horizontalalignment = 'left', verticalalignment = 'bottom')



ax.set_ylabel('Areal Discharge Capacity \n[mAh/cm$^2$]')
ax.set_xlabel('Thickness [$\mu m$]')

fig.tight_layout()
plt.show()