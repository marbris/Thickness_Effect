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
styles = ['-', '--', ':', '.-']

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


ax.set_ylim((ymin, ymax))
ax.set_xlim((xmin, xmax))


#%%

# Cols2 = ['#e76f51', '#f4a261', '#e9c46a', '#2a9d8f', '#264653']

Cols2 = ['#e9c46a', '#e76f51', '#2a9d8f',  '#264653']
markers = ['o', 'v', 's', '^', 'd']
styles = ['-', '--', ':', '.-']

def colfun2(i):
    return Cols2[np.mod(i,len(Cols2))]


def markerfun(i):
    return markers[np.mod(i,len(markers))]

def stylefun(i):
    return styles[np.mod(i,len(styles))]

def sizefun(t):
    #k=0.06
    #k=0.03
    #return Thickness*k
    return 4

xLab = 'Cathode Thickness [$\mu$m]'
#xLab = '"real" Current Density [mA/cm$^2$]'
yLab = 'Cathode Utilization [%]'

xmin = 1
xmax = 400

#xmin = 1e-4
#xmax = 1e-1

ymin = 0
ymax = 105


Leg_kwargs = {'loc': 'upper right'}


fig, ax= plt.subplots(figsize=(Col1,Col1*AR*1.2))    



def Utilization(L,Ld):
    Ut = Ld/L*100
    Ut[Ut>100] = 100
    return Ut

xx = np.linspace(xmin,xmax,int(1e3))

Ldlist = [100, 150, 200]


for i, Ld in enumerate(Ldlist):
    
    plt_kwargs = { 'color' : colfun2(i), 
                    'linestyle' : stylefun(i),
                    'linewidth' : 1 
                    }
    
    
    yy = Utilization(xx,Ld)
    plt.plot(xx,yy,**plt_kwargs)

# ax.annotate("Annotation",
#             xy=(10, 90), xycoords='data',
#             xytext=(10, 10), textcoords='offset points',
#             )

ax.annotate("",
            fontsize=12,
            xy=(110, 30), xycoords='data',
            xytext=(300, 90), textcoords='data',
            
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=-0.1",
                            color = '0.4'),
            ) 

plt.text(170, 15, 'Increasing \ncurrent density', fontsize=10)

ax.set_ylim((ymin, ymax))
ax.set_xlim((xmin, xmax))

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, 
          ncol = 1,#NCol, 
          framealpha = 0, 
          columnspacing=0.7, 
          handletextpad = 0.3, 
          labelspacing = 0.3,
          **Leg_kwargs)


ax.set_ylabel(yLab)
ax.set_xlabel(xLab)
fig.tight_layout()
plt.show()



#%%



# Cols2 = ['#e76f51', '#f4a261', '#e9c46a', '#2a9d8f', '#264653']

Cols2 = ['#e9c46a', '#e76f51', '#2a9d8f',  '#264653']
markers = ['o', 'v', 's', '^', 'd']
styles = ['-', '--', ':', '.-']

def colfun2(i):
    return Cols2[np.mod(i,len(Cols2))]


def markerfun(i):
    return markers[np.mod(i,len(markers))]

def stylefun(i):
    return styles[np.mod(i,len(styles))]

def sizefun(t):
    #k=0.06
    #k=0.03
    #return Thickness*k
    return 4

xLab = 'Cathode Thickness [$\mu$m]'
#xLab = '"real" Current Density [mA/cm$^2$]'
yLab = 'Cathode Utilization [%]'

xmin = 1
xmax = 400

#xmin = 1e-4
#xmax = 1e-1

ymin = 0
ymax = 105

Leg_kwargs = {'loc': 'upper right'}


fig, ax= plt.subplots(figsize=(Col1,Col1*AR*1.2))    



def Utilization(L,Ld):
    Ut = Ld/L*100
    Ut[Ut>100] = 100
    return Ut

xx = np.linspace(xmin,xmax,int(1e3))

Ldlist = [100, 150, 200]


for i, Ld in enumerate(Ldlist):
    
    plt_kwargs = { 'color' : colfun2(i), 
                    'linestyle' : stylefun(i),
                    'linewidth' : 1 
                    }
    
    
    yy = Utilization(xx,Ld)
    plt.plot(xx,yy,**plt_kwargs)

# ax.annotate("Annotation",
#             xy=(10, 90), xycoords='data',
#             xytext=(10, 10), textcoords='offset points',
#             )

ax.annotate("",
            fontsize=12,
            xy=(110, 30), xycoords='data',
            xytext=(300, 90), textcoords='data',
            
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=-0.1",
                            color = '0.4'),
            ) 

plt.text(170, 15, 'Increasing \ncurrent density', fontsize=10)

ax.set_ylim((ymin, ymax))
ax.set_xlim((xmin, xmax))

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, 
          ncol = 1,#NCol, 
          framealpha = 0, 
          columnspacing=0.7, 
          handletextpad = 0.3, 
          labelspacing = 0.3,
          **Leg_kwargs)


ax.set_ylabel(yLab)
ax.set_xlabel(xLab)
fig.tight_layout()
plt.show()