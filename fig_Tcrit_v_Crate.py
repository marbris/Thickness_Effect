

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
                            



dfall, dfCrit, dfOPCrit = main.dfall_dfCrit()


                    

#%% Critical Thickness vs Crate double column figure

Cols = ['#e36414','#0db39e','#748cab','#8a5a44','#264653','#e9c46a']
#Cols2 = ['#e76f51', '#f4a261', '#e9c46a', '#2a9d8f', '#264653']
markers = ['o','s','d', 'v', '^', '>']
lines = ['-',':','--','-.', '','-']
def stylefun(G):
    return (Cols[G], markers[G], lines[G])

fig, ax= plt.subplots(figsize=(Col2,Col1*AR*1.2))    

Lfun = lambda R, k: np.sqrt(k/R)
Rfun = lambda L, k: k/L**2 

xmin = 1e-1
xmax = 2.1e1
ymin = -10
ymax = 300

ax.set_ylim((ymin, ymax))
ax.set_xlim((xmin, xmax))

"""
xx = np.logspace(np.log10(xmin),np.log10(xmax), 100)

kkcms = np.array([2**i for i in range(-1,7,2)])*1e-8
kkmuh = kkcms*3.6e11

for i, k in enumerate(kkmuh):
    ax.plot(xx, Lfun(xx,k), linestyle=':', color=[0.5,0.5,0.5], linewidth = 1)
    
    if Rfun(ymax, k) < xmin:
        xpos = xmin*1.02
        ypos = Lfun(xpos, k)-11
    else:
        ypos = ymax-2
        xpos = Rfun(ymax, k)*0.9
        
    #print((xpos,ypos))
    if i != 0:
        #ypos = 200
        #xpos = Rfun(ypos, k)*0.87
        ax.text(xpos, ypos, "{:.0f}".format(kkcms[i]/1e-8), color = 'k', fontsize =8, horizontalalignment = 'left', verticalalignment = 'top', rotation = -40)
    else:
        #xpos = 1.1e-1
        #ypos = Lfun(xpos, k)-10
        ax.text(xpos, ypos, "RL$_d^2$ = {}".format(kkcms[i]/1e-8) +  " [$10^{-8}$ cm$^2$/s]", color = 'k', fontsize = 8, horizontalalignment = 'left', verticalalignment = 'top', rotation = -20)

"""
Cc = dfCrit.loc[dfCrit['Batch']=='211202_NMC', 'C-rate(prog)']
index2 = (Cc > 0.1) & (Cc < 4.8) 
ms = 4
lw = 1.5
for i, (name, group) in enumerate(dfCrit.groupby(by=['Batch'])):
    
    df = group
    print(name)
    
    index = ~df['Thickness_lo(um)'].isnull() & ~df['Thickness_hi(um)'].isnull()
    
    if name == '211202_NMC': 
        index = index & index2
    
    
    label = name
    
    xx = np.array(df.loc[index,'C-rate_mean(1/h)'].tolist())
    yy = np.array(df.loc[index,'Thickness_max(um)'].tolist())
    
    rl2 = xx*yy**2/3.6e11
    #print("name: {}, rl2_mean(cm2/s): {}, rl2_std(cm2/s): {}".format(name, rl2.mean(), rl2.std()))
    lo = np.array(df.loc[index,'Thickness_lo(um)'].tolist())
    hi = np.array(df.loc[index,'Thickness_hi(um)'].tolist())
    ee = np.vstack([lo,hi])
    ax.errorbar(xx, yy, yerr = ee, 
                marker = stylefun(i)[1], 
                linestyle = stylefun(i)[2], 
                color = stylefun(i)[0], 
                capsize = 3, 
                label = label, 
                markersize=ms, 
                linewidth = lw)


#here I add a few simulated points

Sim_Crate = [1.25, 1.5, 1.75, 2.0]
Sim_CritT = [250.0, 250.0, 200, 200]
lo = [25,25,25,25]
hi = [25,25,25,25]
ee = np.vstack([lo,hi])
ax.errorbar(Sim_Crate, Sim_CritT, yerr = ee, 
                marker = 'o', 
                linestyle = '-', 
                color = 'k', 
                capsize = 3, 
                label = 'Simulated Li/LFP', 
                markersize=ms, 
                linewidth = lw)



ax.set_ylabel('Penetration Depth \nL$_d$ [$\mu m$]')
ax.set_xlabel('C-rate, R [h$^{-1}$]')

ax.set_xscale('log')


handles, labels = ax.get_legend_handles_labels()
"""
ax.legend(handles, labels, 
          ncol = 2, 
          framealpha = 0.7, 
          fontsize = 8,
          #bbox_to_anchor=(-0.01,-0.03), 
          loc = 'upper right',
          columnspacing=0.5, 
          handletextpad = 0.3, 
          labelspacing = 0.3)
"""
ax.legend(handles, labels, 
          ncol = 1, 
          framealpha = 0, 
          fontsize = 8,
          #bbox_to_anchor=(-0.01,-0.03), 
          loc = 'lower left',
          columnspacing=0.5, 
          handletextpad = 0.3, 
          labelspacing = 0.3)



fig.tight_layout()
plt.show()