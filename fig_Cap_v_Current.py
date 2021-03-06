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



#dfplot = dfall.loc[dfall['Batch']=='211202_NMC'].copy()

Samples = ['06', '08', '12', '16', '19']

index = (dfall['Batch']=='211202_NMC') & dfall['Sample'].isin(Samples)
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


Settings = 2



if Settings==0:
    # Area Cap vs C-rate
    xCol = 'C-rate(1/h)'
    yCol = 'Discharge_Capacity(mAh/cm2)'
    
    xLab = 'C-rate [h$^{-1}$]'
    yLab = 'Discharge Capacity \n[mAh/cm$^2$]'
    
    #xmin = 9e-2
    xmin = 9e-2
    xmax = 2.2e1    
    ymin = -0.2
    ymax = 8
    
    
    Leg_kwargs = {'loc': 'upper right', 'bbox_to_anchor' : (1.01, 1.05)}
    Leg_Dummy_tt = -1
    Leg_Col_Order = -1
    Leg_Row_Order = -1
    
elif Settings==1:
    # AMmass Cap vs C-rate
    xCol = 'C-rate(1/h)'
    yCol = 'Discharge_Capacity(mAh/gAM)'
    
    xLab = 'C-rate [h$^{-1}$]'
    yLab = 'Discharge Capacity \n[mAh/g$_{AM}$]'
    
    xmin = 9e-2
    xmax = 2.2e1
    ymin = 0
    ymax = 250
    
    Leg_kwargs = {'loc': 'lower left', 'bbox_to_anchor' : (-0.01, -0.05)}
    Leg_Dummy_tt = -1
    Leg_Col_Order = 1
    Leg_Row_Order = 1
    
elif Settings==2:
    # Area Cap vs Current density
    xCol = 'Discharge_Current(mA/cm2)'
    yCol = 'Discharge_Capacity(mAh/cm2)'
    
    xLab = 'Current Density [mA/cm$^2$]'
    yLab = 'Discharge Capacity \n[mAh/cm$^2$]'
    
    xmin = 1e-1
    xmax = 2e2
    
    ymin = -0.2
    ymax = 8
    
    Leg_kwargs = {'loc': 'upper right'}
    Leg_Dummy_tt = -1
    Leg_Col_Order = -1
    Leg_Row_Order = -1
    
elif Settings==3:
    # AMmass Cap vs Current density
    xCol = 'Discharge_Current(mA/cm2)'
    yCol = 'Discharge_Capacity(mAh/gAM)'
    
    xLab = 'Current Density [mA/cm$^2$]'
    yLab = 'Discharge Capacity \n[mAh/g$_{AM}$]'
    
    xmin = 1e-1
    xmax = 2e2
    ymin = 0
    ymax = 250
    
    Leg_kwargs = {'loc': 'lower left'}
    Leg_Dummy_tt = -1
    Leg_Col_Order = 1
    Leg_Row_Order = 1


fig, ax= plt.subplots(figsize=(Col2,Col1*AR*1.2))    
#fig, ax= plt.subplots(figsize=(13,6))  




ax.set_ylim((ymin, ymax))
ax.set_xlim((xmin, xmax))

SampleList = dfplot.loc[:,'Sample'].dropna().unique().tolist()
#plotting the samples in reverse so that thickest are at the bottom
PlotList = [SampleList[-i] for i in range(len(SampleList))]

ttss=np.array([])
wtt=np.array([])
for i, SampleID in enumerate(PlotList):
    
    df = dfplot.loc[dfplot['Sample']==SampleID, :]
    Thickness = df.loc[:,'Thickness(um)'].unique()[0]
    WT = df.loc[:,'Wet_Thickness(um)'].unique()[0]
    
    ttss = np.append(ttss, Thickness+int(SampleID)*1e-2)
    wtt = np.append(wtt, WT)
    
    df.plot(x = xCol, 
            y = yCol, 
            ax=ax, 
            color = colfun(WT), 
            marker = markerfun(i),#markerfun(WT,ttss[-1]), 
            logx=True, 
            markersize = sizefun(Thickness), 
            label = '{:.0f} $\mu$m'.format(Thickness),
            zorder = 200-Thickness)

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


ii = np.logspace(0.0, 2.0, int(1e2))
#ccap = 13.5*np.exp(-ii*0.11)
ccap = 700*np.exp(-ii*0.13)

#plt.plot(ii,ccap,marker='',color='k', linestyle=':')

ax.set_xscale('log')



ax.set_ylabel(yLab)
ax.set_xlabel(xLab)
fig.tight_layout()
plt.show()



#%%




batchlabel = 'D.Y.W. Yu (2006), LFP'
#batchlabel = 'H. Zheng (2012), NMC'
#batchlabel = 'M. Singh (2016), NMC'
#batchlabel = '220203_NMC'
#batchlabel = '211202_NMC'
#batchlabel = 'C. Heubner (2020), LTO'
dfplot = dfall.loc[dfall['Batch']==batchlabel].copy()

#I'm dropping the final few cycles, since those repeat the low-C cycles.
dfplot.drop(dfplot.loc[(dfall['C-rate(1/h)']<0.3) & (dfplot['Cycle']>2)].index, inplace=True)

#One of the 220203_NMC cycles are weird. I'm just removing it. 
#index = dfplot.loc[(dfplot['Cycle']==2) & (dfplot['Thickness(um)'] == 198)].index
#dfplot.drop(index, inplace=True)

Cols2 = ['#e76f51', '#f4a261', '#e9c46a', '#2a9d8f', '#264653']
markers = ['o', 'v', 's', '^']



def sizefun(t):
    #k=0.06
    #k=0.03
    #return Thickness*k
    return 4

def colfun2(i):
    return Cols2[np.mod(i,len(Cols2))]

def markfun2(i):
    return markers[np.mod(i,len(markers))]


Settings = 3



if Settings==0:
    # Area Cap vs C-rate
    xCol = 'Avg_C-rate(1/h)'
    yCol = 'Avg_DCapacity(mAh/cm2)'
    
    xLab = 'C-rate [h$^{-1}$]'
    yLab = 'Discharge Capacity \n[mAh/cm$^2$]'
    
    #xmin = 9e-2
    xmin = 9e-2
    xmax = 1.1e0    
    ymin = -0.2
    ymax = 8
    
    Leg_kwargs = {'loc': 'lower left', 'bbox_to_anchor' : (-0.01, -0.05)}
    Leg_Dummy_tt = -1
    Leg_Col_Order = -1
    Leg_Row_Order = -1
    
elif Settings==1:
    # AMmass Cap vs C-rate
    xCol = 'C-rate(1/h)'
    yCol = 'Capacity(mAh/gAM)'
    
    xLab = 'C-rate [h$^{-1}$]'
    yLab = 'Discharge Capacity \n[mAh/g$_{AM}$]'
    
    xmin = 9e-2
    xmax = 2.2e1
    ymin = 0
    ymax = 250
    
    Leg_kwargs = {'loc': 'lower left', 'bbox_to_anchor' : (-0.01, -0.05)}
    Leg_Dummy_tt = -1
    Leg_Col_Order = 1
    Leg_Row_Order = 1
    
elif Settings==2:
    # Area Cap vs Current density
    xCol = 'Discharge_Current(mA/cm2)'
    yCol = 'Discharge_Capacity(mAh/cm2)'
    
    xLab = 'Current Density [mA/cm$^2$]'
    yLab = 'Discharge Capacity \n[mAh/cm$^2$]'
    
    xmin = 1e-1
    xmax = 2e2
    
    ymin = -0.2
    ymax = 4
    
    Leg_kwargs = {'loc': 'upper right'}
    Leg_Dummy_tt = -1
    Leg_Col_Order = -1
    Leg_Row_Order = -1
    
elif Settings==3:
    # AMmass Cap vs Current density
    xCol = 'Discharge_Current(mA/cm2)'
    yCol = 'Discharge_Capacity(mAh/gAM)'
    
    xLab = 'Current Density [mA/cm$^2$]'
    yLab = 'Discharge Capacity \n[mAh/g$_{AM}$]'
    
    xmin = 1e-1
    xmax = 2e2
    ymin = 0
    ymax = 250
    #ymax = 0.2
    
    Leg_kwargs = {'loc': 'upper right'}
    Leg_Dummy_tt = -1
    Leg_Col_Order = 1
    Leg_Row_Order = 1


fig, ax= plt.subplots(figsize=(Col2,Col1*AR*1.2))   
fig, ax= plt.subplots(figsize=(Col2,Col1*AR*1.2))    
#fig, ax= plt.subplots(figsize=(13,6))  




ax.set_ylim((ymin, ymax))
ax.set_xlim((xmin, xmax))

SampleList = dfplot.loc[:,'Sample'].dropna().unique().tolist()
#plotting the samples in reverse so that thickest are at the bottom
PlotList = [SampleList[-i] for i in range(len(SampleList))]

ttss=np.array([])
wtt=np.array([])
for i, SampleID in enumerate(PlotList):
    
    df = dfplot.loc[dfplot['Sample']==SampleID, :]
    Thickness = df.loc[:,'Thickness(um)'].unique()[0]
    WT = df.loc[:,'Wet_Thickness(um)'].unique()[0]
    
    ttss = np.append(ttss, Thickness+int(SampleID)*1e-2)
    wtt = np.append(wtt, WT)
    
    df.plot(x = xCol, 
            y = yCol, 
            ax=ax, 
            #color = colfun(WT), 
            color = colfun2(i), 
            marker = markfun2(i), 
            logx=True, 
            markersize = sizefun(Thickness), 
            label = '{:.0f}'.format(Thickness),
            zorder = 200-Thickness)

ax.set_xscale('log')

plt.text(0.12,50,batchlabel, fontsize=16)

# NCol = len(np.unique(wtt))
# maxrows = 4
# for wti in np.unique(wtt):
#     N_dummies = maxrows - len(wtt[wtt==wti])
#     for i in range(N_dummies):
#             ax.plot(np.zeros(1), np.zeros([1,2]), color='w', alpha=0, label=' ')
#             wtt = np.append(wtt, wti)
#             ttss = np.append(ttss, Leg_Dummy_tt)


handles, labels = ax.get_legend_handles_labels()
#labels, handles, ttss2, wtt2 = zip(*sorted(zip(labels, handles, ttss, wtt), key=lambda k: (Leg_Col_Order*k[3], Leg_Row_Order*k[2]), reverse=False))
labels, handles, ttss2 = zip(*sorted(zip(labels, handles, ttss), key=lambda k: k[2], reverse=False))
ax.legend(handles, labels, 
          ncol = 2, 
          framealpha = 0, 
          columnspacing=0.7, 
          handletextpad = 0.3, 
          labelspacing = 0.3,
          **Leg_kwargs)


ax.set_ylabel(yLab)
ax.set_xlabel(xLab)
fig.tight_layout()
plt.show()
# %%


#%%




#batchlabel = 'D.Y.W. Yu (2006), LFP'
#batchlabel = 'H. Zheng (2012), NMC'
#batchlabel = 'M. Singh (2016), NMC'
#batchlabel = '220203_NMC'
batchlabel = '211202_NMC'
Samples = ['05', '09', '12', '15', '17', '19']

#batchlabel = 'C. Heubner (2020), NMC'
#Samples = ['119', '129', '163', '166', '206', '216']


index = (dfall['Batch']==batchlabel) & dfall['Sample'].isin(Samples)
dfplot = dfall.loc[index].copy()

#I'm dropping the final few cycles, since those repeat the low-C cycles.
dfplot.drop(dfplot.loc[(dfall['C-rate(1/h)']<0.3) & (dfplot['Cycle']>2)].index, inplace=True)

#One of the 220203_NMC cycles are weird. I'm just removing it. 
#index = dfplot.loc[(dfplot['Cycle']==2) & (dfplot['Thickness(um)'] == 198)].index
#dfplot.drop(index, inplace=True)

Cols2 = ['#e76f51', '#f4a261', '#e9c46a', '#2a9d8f', '#264653']
markers = ['o', 'v', 's', '^']



def sizefun(t):
    #k=0.06
    #k=0.03
    #return Thickness*k
    return 4

def colfun2(i):
    return Cols2[np.mod(i,len(Cols2))]

def markfun2(i):
    return markers[np.mod(i,len(markers))]


Settings = 3

# AMmass Cap vs Current density
xCol = 'Discharge_Current(mA/cm2)'
yCol = 'Discharge_Capacity(mAh/gAM)'

xLab = 'Current Density [mA/cm$^2$]'
yLab = 'Discharge Capacity \n[mAh/g$_{AM}$]'

# xmin = 1e-1
# xmax = 2e2
# ymin = 0
# ymax = 250
#ymax = 0.2

xmin = 1e-1
xmax = 1e2
ymin = 0
#ymax = 170
ymax = 240

Leg_kwargs = {'loc': 'lower left',
              'ncol': 1,
              'fontsize': 10}



fig, ax= plt.subplots(figsize=(Col1,Col1*AR*1.2))    
#fig, ax= plt.subplots(figsize=(13,6))  




ax.set_ylim((ymin, ymax))
ax.set_xlim((xmin, xmax))

SampleList = dfplot.loc[:,'Sample'].dropna().unique().tolist()
#plotting the samples in reverse so that thickest are at the bottom
PlotList = [SampleList[-i] for i in range(len(SampleList))]

ttss=np.array([])
wtt=np.array([])
for i, SampleID in enumerate(PlotList):
    
    df = dfplot.loc[dfplot['Sample']==SampleID, :]
    Thickness = df.loc[:,'Thickness(um)'].unique()[0]
    WT = df.loc[:,'Wet_Thickness(um)'].unique()[0]
    
    ttss = np.append(ttss, Thickness+int(SampleID)*1e-2)
    wtt = np.append(wtt, WT)
    
    df.plot(x = xCol, 
            y = yCol, 
            ax=ax, 
            #color = colfun(WT), 
            color = colfun2(i), 
            marker = markfun2(i), 
            logx=True, 
            markersize = 4, 
            linewidth=1,
            label = '{:.0f} $\mu$m'.format(Thickness),
            zorder = 200-Thickness)

ax.set_xscale('log')
#textstring = batchlabel
textstring = 'Present Study, NMC'
plt.text(xmax*0.93,ymax-2,textstring, fontsize=8, horizontalalignment='right', verticalalignment='top')

#plt.text(0.9,138,'$\mu$m', fontsize=13, horizontalalignment='right', verticalalignment='top')

# NCol = len(np.unique(wtt))
# maxrows = 4
# for wti in np.unique(wtt):
#     N_dummies = maxrows - len(wtt[wtt==wti])
#     for i in range(N_dummies):
#             ax.plot(np.zeros(1), np.zeros([1,2]), color='w', alpha=0, label=' ')
#             wtt = np.append(wtt, wti)
#             ttss = np.append(ttss, Leg_Dummy_tt)


handles, labels = ax.get_legend_handles_labels()
#labels, handles, ttss2, wtt2 = zip(*sorted(zip(labels, handles, ttss, wtt), key=lambda k: (Leg_Col_Order*k[3], Leg_Row_Order*k[2]), reverse=False))
labels, handles, ttss2 = zip(*sorted(zip(labels, handles, ttss), key=lambda k: k[2], reverse=False))
ax.legend(handles, labels, 
          framealpha = 0, 
          columnspacing=0.7, 
          handletextpad = 0.3, 
          labelspacing = 0.3,
          **Leg_kwargs)


ax.set_ylabel(yLab)
ax.set_xlabel(xLab)
fig.tight_layout()
plt.show()
# %%
