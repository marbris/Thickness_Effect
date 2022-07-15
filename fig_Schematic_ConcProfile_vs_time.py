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


fun = lambda x, b: (1-math.erf(x/5))/2*(1-b) + b

def erffun(x,b):
    yy = np.array([fun(i, b) for i in x])
    return yy


fig, ax= plt.subplots(figsize=(Col1,Col1*AR*1.2)) 

xmin=0
xmax=100

ymin=0
ymax=1.1

xx = np.linspace(xmin, xmax,1000)


bb = [0.3,0.15,0.05,0.01]
tt = [0,-1,2,10]

for i, b in enumerate(bb):
    if i==0:
        yy = np.ones(xx.shape)*b
    else:
        yy = erffun(xx-tt[i],b)
    
    plt.plot(xx,yy, 
             color = colfun(i),
             linestyle = stylefun(i))
    
    if i==3:
        plt.text(xmax+2, yy[-1], 't$_{:.0f}$'.format(i), fontsize=10, verticalalignment = 'top')
    else:
        plt.text(xmax+2, yy[-1], 't$_{:.0f}$'.format(i), fontsize=10, verticalalignment = 'center')



ax.set_xticks([])
ax.set_yticks([])

ax.set_ylim((ymin, ymax))
ax.set_xlim((xmin, xmax))

ax.set_ylabel('Li$^+$ conc. in solution')
ax.set_xlabel('Distance from Separator')




for i in range(3):
    ypos = 0.5 + 0.15*i
    ax.annotate("",
                xy=(15, ypos), xycoords='data',
                xytext=(0, ypos), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3"),
                )
    
plt.text(15, ypos+0.05, 'Li$^+$ diffusing \ninto cathode', fontsize=10)


for i in range(4):
    xpos = 25+i*20
    ax.annotate("",
                xy=(xpos, 0.2), xycoords='data',
                xytext=(xpos, 0.4), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3"),
                )


plt.text(22, 0.45, 'Li$^+$ consumed by intercalation', fontsize=10)



fig.tight_layout()
plt.show()

#%%
