#%%

import Cycler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import main
from scipy.interpolate import interp1d

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

plt.rc('figure', dpi = 100)

Col1= 8.9/2.54 # 8.9cm single column figure width in Nature
Col2= 18.3/2.54 # 18.3cm double column figure width in Nature

AR = 1/1.618

dfall, dfCapCrit, dfOPCrit = main.dfall_dfCrit()



#%%



#dfplot = dfall.loc[dfall['Batch']=='211202_NMC'].copy()

Samples = ['06', '08', '12', '16', '18']

index = (dfall['Batch']=='211202_NMC') # & dfall['Sample'].isin(Samples)
dfplot = dfall.loc[index].copy()

#I'm dropping the final few cycles, since those repeat the low-C cycles.
dfplot.drop(dfplot.loc[(dfall['C-rate(1/h)']<0.3) & (dfplot['Cycle']>2)].index, inplace=True)

Cols2 = ['#e76f51', '#f4a261', '#e9c46a', '#2a9d8f', '#264653']
markers = ['o', 'v', 's', '^', 'd']

def colfun(wt):
    if wt==600:
        col = Cols2[0]
    elif wt==500:
        col = Cols2[1]
    elif wt==400:
        col = Cols2[2]
    elif wt==300:
        col = Cols2[3]
    elif wt==200:
        col = Cols2[4]
    else:
        'k'
    
    return col

#def markerfun(wt, ttss):
    
    
#    dftemp = dfplot.loc[dfplot['Wet_Thickness(um)']==wt, ('Thickness(um)', 'Sample')].drop_duplicates()
    
#    ss_arr = np.array(list(map(int, dftemp.loc[:,'Sample'].tolist())))
#    tt_arr = np.array(dftemp.loc[:,'Thickness(um)'].tolist())
    
#    ttss_arr = tt_arr + ss_arr*1e-2
    #the number of thicker samples within this wet thickness
#    N_Thicker = sum(ttss_arr>ttss)
    #print((ttss_arr, ttss, N_Thicker))

    #the marker is determined by how many thicker samples there are
#    mark = markers[N_Thicker]
    
#    return mark

def markerfun(i):
    return markers[np.mod(i,len(markers))]

def sizefun(t):
    #k=0.06
    #k=0.03
    #return Thickness*k
    return 4


# AMmass Cap vs Current density
xCol = 'Discharge_Current(mA/cm2)'
yCol = 'Avg_DCapacity(mAh/gAM)'

xLab = 'Current Density [mA/cm$^2$]'
yLab = 'Cathode Utilization [%]'

xmin = 1e-1
xmax = 2e2
ymin = 0
ymax = 105

Leg_kwargs = {'loc': 'lower left'}
Leg_Dummy_tt = -1
Leg_Col_Order = 1
Leg_Row_Order = 1


#fig, ax= plt.subplots(figsize=(Col2,Col1*AR*1.2))    
fig, ax= plt.subplots(figsize=(10,6))  




ax.set_ylim((ymin, ymax))
ax.set_xlim((xmin, xmax))

SampleList = dfplot.loc[:,'Sample'].dropna().unique().tolist()
#plotting the samples in reverse so that thickest are at the bottom
PlotList = [SampleList[-i] for i in range(len(SampleList))]

ttss=np.array([])
wtt=np.array([])
tt = np.array([])

Ilist = [5,10,15,20]
Ut = pd.DataFrame(columns = ['Utilization[%]', 'Current[mA/cm2]', 'Thickness[um]'])

for i, (Sample, df) in enumerate(dfplot.groupby('Sample')):
    
    Thickness = df.loc[:,'Thickness(um)'].unique()[0]
    WT = df.loc[:,'Wet_Thickness(um)'].unique()[0]
    
    ttss = np.append(ttss, Thickness+int(SampleID)*1e-2)
    wtt = np.append(wtt, WT)
    
    C01 = df.loc[df['Cycle']==2,'Avg_DCapacity(mAh/gAM)'].to_numpy(dtype=float)[0]
    
    Y = df.loc[:,'Avg_DCapacity(mAh/gAM)'].to_numpy(dtype=float)/C01*100
    X = df.loc[:,'Discharge_Current(mA/cm2)'].to_numpy(dtype=float)
    
    plt_kwargs = {  #'ax' : ax, 
                    'color' : colfun(WT), 
                    'marker' : markerfun(i),
                    #'logx': True, 
                    'markersize' : sizefun(Thickness), 
                    'label' :'{:.0f} $\mu$m'.format(Thickness),
                    'zorder' : 200-Thickness}
    
    plt.plot(X,Y,**plt_kwargs)
    
    f = interp1d(X,Y)
    
    for i in Ilist:
        data = {'Utilization[%]': f(i),
                'Current[mA/cm2]': i,
                'Thickness[um]': Thickness}
        dftemp = pd.DataFrame(data=data, index = [0])
        Ut = pd.concat([Ut, dftemp], ignore_index=True)
    #        y = yCol, 
    #        **plt_kwargs)
    
    
Ut = Ut.sort_values(by=['Current[mA/cm2]', 'Thickness[um]'])

ax.set_xscale('log')


"""
NCol = len(np.unique(wtt))
maxrows = 4
for wti in np.unique(wtt):
    N_dummies = maxrows - len(wtt[wtt==wti])
    for i in range(N_dummies):
            ax.plot(np.zeros(1), np.zeros([1,2]), color='w', alpha=0, label=' ')
            wtt = np.append(wtt, wti)
            ttss = np.append(ttss, Leg_Dummy_tt)
"""

handles, labels = ax.get_legend_handles_labels()
labels, handles, ttss2, wtt2 = zip(*sorted(zip(labels, handles, ttss, wtt), key=lambda k: (Leg_Col_Order*k[3], Leg_Row_Order*k[2]), reverse=False))
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



fig, ax= plt.subplots(figsize=(10,6))  

index = np.argsort(tt)
Ut_10a = Ut_10a[index]
tt = tt[index]

for I, df in Ut.groupby(by=['Current[mA/cm2]']):
    
    plt_kwargs = {  'ax' : ax, 
                    #'color' : colfun(WT), 
                    'marker' : markerfun(i),
                    #'logx': True, 
                    #'markersize' : sizefun(Thickness), 
                    'label' :'{:.0f} mA/cm$^2$'.format(I)}
    
    df.plot(x='Thickness[um]', y = 'Utilization[%]', **plt_kwargs)

#plt.plot(tt,Ut_10a,marker = '.',)



ax.set_xlabel('Cathode Thickness [$\mu$m]')
ax.set_ylabel('Utilization [%]')