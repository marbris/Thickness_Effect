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



fig, ax = plt.subplots(figsize=(Col2,5)) 
# ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
# ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
# ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
# ax4 = plt.subplot2grid((3, 3), (2, 0))
# ax5 = plt.subplot2grid((3, 3), (2, 1))


# ax1.set_position((0,0,0.5,0.5))

# ax = [ax1,ax2,ax3,ax4,ax5]


xmin=0
xmax=100

ymin=0
ymax=1.2

xx = np.linspace(xmin, xmax,1000)

ttt = [25,50,75]

for i in range(3):
    
    yy = SolidCfun(xx,35)
    ax[i].fill_between(xx,yy, 
            color = 'k',
            linestyle = '-', 
            alpha = 0.3)

    
    ax[i].set_xticks([])
    ax[i].set_yticks([])

    ax[i].set_ylim((ymin, ymax))
    ax[i].set_xlim((xmin, xmax))
    
fig.tight_layout()
plt.show()
    
#%%
        