#%%

import Cycler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import main
from scipy.interpolate import interp1d

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 15

Axes = {'titlesize': SMALL_SIZE,    # fontsize of the axes title
        'labelsize': MEDIUM_SIZE}   # fontsize of the x and y labels
plt.rc('axes', **Axes)

#plt.rc('text', usetex=True)

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

plt.rc('figure', dpi = 200)

Col1= 8.9/2.54 # 8.9cm single column figure width in Nature
Col2= 18.3/2.54 # 18.3cm double column figure width in Nature

AR = 1/1.618



#%%



Cols = ['k','#e9c46a', '#e76f51', '#2a9d8f',  '#264653']
styles = ['--', '-.', ':', '-']

def colfun(i):
    return Cols[np.mod(i,len(Cols))]

def stylefun(i):
    return styles[np.mod(i,len(styles))]

erffun = lambda x, t: (1-math.erf((x-t)/6))/2

def SolidCfun(x,t):
    yy = np.array([erffun(i,t) for i in x])
    return yy

tt = 40

LiCfun = lambda x,t: (1-x/90)*SolidCfun(x,t)
Reacfun = lambda x,t: LiCfun(x,t)*(1-SolidCfun(x,t))*5

fig, ax = plt.subplots(3,1,figsize=(Col2,5)) 

xmin=0
xmax=100

ymin=0
ymax=1.2

xx = np.linspace(xmin, xmax,1000)

ttt = [25,50,75]

for i in range(3):
    
    tt = ttt[i]
    
    yy = SolidCfun(xx,tt)
    ax[i].fill_between(xx,yy, 
            color = 'k',
            linestyle = '-', 
            alpha = 0.3)

    yy = LiCfun(xx,tt)
    ax[i].fill_between(xx,yy, 
            color = Cols[3],
            linestyle = '-', 
            alpha = 0.5)

    yy = Reacfun(xx,tt)
    ax[i].fill_between(xx,yy, 
            color = Cols[2],
            linestyle = '-',
            alpha = 0.6)
    
    ax[i].set_xticks([])
    ax[i].set_yticks([])

    ax[i].set_ylim((ymin, ymax))
    ax[i].set_xlim((xmin, xmax))
    
    
    
    
        
#ax[0].text(24, 1.0, 'Intercalated Li$^+$', fontsize=10, verticalalignment = 'center', color = 'k')

ax[0].annotate("Intercalated Li$^+$",
                xy=(20, 0.87), xycoords='data',
                xytext=(40, 0.60), textcoords='data',
                arrowprops=dict(arrowstyle="-|>",
                                connectionstyle="arc3,rad=0.3",
                                relpos=(0,0.7),
                                color='k'))


#ax[0].text(1, 0.5, 'Li$^+$ in Electrolyte', fontsize=10, verticalalignment = 'center', color = 'w')

ax[0].annotate('Li$^+$ in Electrolyte',
                xy=(8, 0.88), xycoords='data',
                xytext=(1, 0.3), textcoords='data',
                color='w',
                arrowprops=dict(arrowstyle="-|>",
                                connectionstyle="arc3,rad=-0.2",
                                relpos=(0.3,1),
                                color='w'))

#ax[0].text(30, 0.5, '[Li$^+$ in Electrolyte] times \n[Vacant intercalation-sites]', fontsize=10, verticalalignment = 'center', color = Cols[2])

ax[0].annotate('[Li$^+$ in Electrolyte] times \n[Vacant intercalation-sites]',
                xy=(29, 0.45), xycoords='data',
                xytext=(40, 0.10), textcoords='data',
                color=Cols[2],
                arrowprops=dict(arrowstyle="-|>",
                                connectionstyle="arc3,rad=0.2",
                                relpos=(0,0.7),
                                color=Cols[2]))

dttt=8
ypos1 = ttt[0]-dttt
ypos2 = ttt[0]+dttt
ax[0].vlines([ypos1,ypos2],1,ymax,color = 'k')

ax[0].text(ttt[0], 1.1, 'Reaction Zone', fontsize=10, verticalalignment = 'center', horizontalalignment = 'center', color = 'k')
ax[0].text(ypos1-1, 1.1, 'Surface Zone', fontsize=10, verticalalignment = 'center', horizontalalignment = 'right', color = 'k')
ax[0].text(ypos2+1, 1.1, 'Depletion Zone', fontsize=10, verticalalignment = 'center', horizontalalignment = 'left', color = 'k')



ss = 'A steady state Li$^+$ gradient \nemerges in the surface zone, \nwhere no Li$^+$ is consumed'

ax[1].text(1, 0.1, ss, fontsize = 10, verticalalignment = 'bottom', color='w')

xpos = np.array([25 + i*2.0 for i in range(3)])
ypos = LiCfun(xpos,ttt[1])

for i in range(3):
    
    ax[1].annotate("",
                xy=(xpos[i], ypos[i]), xycoords='data',
                xytext=(xpos[i], 0.9), textcoords='data',
                arrowprops=dict(arrowstyle="-|>",
                                connectionstyle="arc3",
                                color='w')
                )

# ax[1].annotate(ss,
#                 xy=(12, 0.84), xycoords='data',
#                 xytext=(1, 0.10), textcoords='data',
#                 color='w',
#                 arrowprops=dict(arrowstyle="-|>",
#                                 connectionstyle="arc3,rad=-0.2",
#                                 relpos=(0.3,1),
#                                 color='w'))

ss = "As the 'intercalation-reactants' \nare dimished, the applied voltage needs \nto increase to maintain the constant current"

ax[1].text(58, 0.10, ss, fontsize = 8, verticalalignment = 'bottom', color=Cols[2])

dttt=2
xpos = np.array([ttt[1] + dttt*(i-1) for i in range(3)])
ypos = Reacfun(xpos,ttt[1])

for i in range(3):
    ax[1].annotate("",
                xy=(xpos[i], ypos[i]), xycoords='data',
                xytext=(xpos[i], 0.78), textcoords='data',
                arrowprops=dict(arrowstyle="-|>",
                                connectionstyle="arc3",
                                color=Cols[2])
                )
    

ss = "The discharge-front grows \nat the rate of the current density"

ax[1].text(47, 0.85, ss, fontsize = 8, verticalalignment = 'bottom', color='k')

#xpos = np.array([ttt[1]-i-3 for i in range(3)])
xpos = np.array([47.5,46,44])
ypos = SolidCfun(xpos,ttt[1])
    
for i in range(3):
    ax[1].annotate("",
                xy=(xpos[i], ypos[i]), xycoords='data',
                xytext=(xpos[0]-7, ypos[i]), textcoords='data',
                arrowprops=dict(arrowstyle="-|>",
                                connectionstyle="arc3",
                                color='k')
                )

ax[1].set_ylabel('Separator')

ss = "At some point, the voltage required\nto maintain the constant current\nexceeds the voltage threshold.\nThe discharge is ended."

ax[2].text(2, 0.1, ss, fontsize = 8, verticalalignment = 'bottom', color='k', horizontalalignment='left',bbox=dict(facecolor='w', alpha=0.5, linestyle=''))


xxmin = ttt[2]-4
xxmax = xmax-1

xx = np.linspace(xxmin, xxmax,100)
yy = SolidCfun(xx,ttt[2]+2) +0.05

ax[2].plot(xx,yy,linestyle=':',color='k')
ax[2].hlines(yy[0],xxmin,xxmax,linestyle=':',color='k')
ax[2].vlines(xxmax,min(yy),max(yy),linestyle=':',color='k')

ss = "Unutilized \nCapacity"

ax[2].text(88, 0.5, ss, fontsize = 12, horizontalalignment='center',verticalalignment = 'center', color='k')

ax[2].set_xlabel('Distance from Separator')




fig.tight_layout()
plt.show()

#%%
