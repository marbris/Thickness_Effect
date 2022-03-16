#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 15:48:21 2022

@author: martin
"""
import Cycler

import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
import dfall_dfCrit

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

#plt.rc('figure', dpi = 200)

Col1= 8.9/2.54 # 8.9cm single column figure width in Nature
Col2= 18.3/2.54 # 18.3cm double column figure width in Nature

AR = 1/1.618
                            


#%%

# SampleList = {
#                 #'220203_NMC': ['10', '11', '15', '16', '24', '25'],
#                 '211202_NMC': ['02', '03', '05', '06', '07', '08', '09', '12', '13', '15', '16', '17', '18', '19']
#              }

# #collecting batch cycle data 
# dfall = Cycler.get_BatchCycles(SampleList)

# #adding HZheng and DYW Yu
# dfnew = Cycler.init_OtherData()

# dfall = dfall.append(dfnew, ignore_index = True)


# #%% creating c-rate groups


# ProgDict = {
#             'Martin_cycles_1': [0.1, 0.5, 0.8, 1.2, 1.8, 2.7, 4.2, 6.4, 9.8, 15.1],
#             'Martin_cycles_2': [0.1, 0.6, 0.9, 1.3, 2.1, 3.2, 4.8, 7.4, 11.3, 17.4],
#             'Martin_cycles_3': [0.1, 0.7, 1, 1.6, 2.4, 3.6, 5.6, 8.5, 13.1, 20],
#             '01C_1C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#             'H. Zheng (2012), NMC': [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50],
#             'H. Zheng (2012), LFP': [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 100, 200, 300],
#             'D.Y.W. Yu (2006), LFP': [0.1, 0.2, 0.5, 1, 2, 5, 8, 12, 18, 25]
#             }

# #here i'm groupding the c-rates within each Batch.
# dfall = dfall.groupby('Batch').apply(Cycler.CRate_groups, ProgDict=ProgDict)

# #%%



# dfCrit = Cycler.get_CritRL(dfall)


dfall, dfCrit = dfall_dfCrit.dfall_dfCrit()


                    
#%% Capacity vs current

dfplot = dfall.loc[dfall['Batch']=='211202_NMC'].copy()

#I'm dropping the final few cycles, since those repeat the low-C cycles.
dfplot.drop(dfplot.loc[(dfall['C-rate(1/h)']<0.3) & (dfplot['Cycle_ID']>2)].index, inplace=True)

Cols2 = ['#e76f51', '#f4a261', '#e9c46a', '#2a9d8f', '#264653']
markers = ['o', 'v', 's', '^']

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
    
    return col

def markerfun(wt, ttss):
    
    
    dftemp = dfplot.loc[dfplot['Wet_Thickness(um)']==wt, ('Thickness(um)', 'SampleID')].drop_duplicates()
    
    ss_arr = np.array(list(map(int, dftemp.loc[:,'SampleID'].tolist())))
    tt_arr = np.array(dftemp.loc[:,'Thickness(um)'].tolist())
    
    ttss_arr = tt_arr + ss_arr*1e-2
    #the number of thicker samples within this wet thickness
    N_Thicker = sum(ttss_arr>ttss)
    #print((ttss_arr, ttss, N_Thicker))

    #the marker is determined by how many thicker samples there are
    mark = markers[N_Thicker]
    
    return mark

def sizefun(t):
    #k=0.06
    #k=0.03
    #return Thickness*k
    return 4


Settings = 1



if Settings==0:
    # Area Cap vs C-rate
    xCol = 'C-rate(1/h)'
    yCol = 'Capacity(mAh/cm2)'
    
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
    xCol = 'Current(mA/cm2)'
    yCol = 'Capacity(mAh/cm2)'
    
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
    xCol = 'Current(mA/cm2)'
    yCol = 'Capacity(mAh/gAM)'
    
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

SampleList = dfplot.loc[:,'SampleID'].dropna().unique().tolist()
#plotting the samples in reverse so that thickest are at the bottom
PlotList = [SampleList[-i] for i in range(len(SampleList))]

ttss=np.array([])
wtt=np.array([])
for SampleID in PlotList:
    
    df = dfplot.loc[dfplot['SampleID']==SampleID, :]
    Thickness = df.loc[:,'Thickness(um)'].unique()[0]
    WT = df.loc[:,'Wet_Thickness(um)'].unique()[0]
    
    ttss = np.append(ttss, Thickness+int(SampleID)*1e-2)
    wtt = np.append(wtt, WT)
    
    df.plot(x = xCol, 
            y = yCol, 
            ax=ax, 
            color = colfun(WT), 
            marker = markerfun(WT,ttss[-1]), 
            logx=True, 
            markersize = sizefun(Thickness), 
            label = '{:.0f}'.format(Thickness),
            zorder = 200-Thickness)


NCol = len(np.unique(wtt))
maxrows = 4
for wti in np.unique(wtt):
    N_dummies = maxrows - len(wtt[wtt==wti])
    for i in range(N_dummies):
            ax.plot(np.zeros(1), np.zeros([1,2]), color='w', alpha=0, label=' ')
            wtt = np.append(wtt, wti)
            ttss = np.append(ttss, Leg_Dummy_tt)


handles, labels = ax.get_legend_handles_labels()
labels, handles, ttss2, wtt2 = zip(*sorted(zip(labels, handles, ttss, wtt), key=lambda k: (Leg_Col_Order*k[3], Leg_Row_Order*k[2]), reverse=False))
ax.legend(handles, labels, 
          ncol = NCol, 
          framealpha = 0, 
          columnspacing=0.7, 
          handletextpad = 0.3, 
          labelspacing = 0.3,
          **Leg_kwargs)


ax.set_ylabel(yLab)
ax.set_xlabel(xLab)
fig.tight_layout()
plt.show()

#%% plotting areacap vs thickness for my sampels

index = dfall['Batch'] == '211202_NMC'
dfplot = dfall.loc[index, ('C-rate(rnd)', 'C-rate(mean)', 'Thickness(um)', 'Avg_DCapacity(mAh/cm2)')].copy()

index = dfCrit['Batch'] == '211202_NMC'
dfCritplot = dfCrit.loc[index, ('C-rate_mean(1/h)', 'C-rate(rnd)', 'Thickness_max(um)','Avg_DCapacity_max(mAh/cm2)')].copy()

fig, ax= plt.subplots(figsize=(Col1,Col1*AR*1.2))    

ax.set_ylim((-0.5, 9))
ax.set_xlim((0, 200))



#crts = np.sort(dfall.loc[:,'C-rate(rnd)'].unique())
#crts2 = crts[crts<=13.1]

crts2 = [0.1, 0.6, 1.0, 1.3,1.6, 2.1, 2.4, 3.6, 5.6, 6.4]

#cr_groups = np.array([0.1, 1.0, 3.6, 13.1])
#Cols = ['black', '#219ebc', '#e36414', '#606c38', 'green']

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
    label = '{:.1f}'.format(df3['C-rate(mean)'].mean())
    
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
          bbox_to_anchor=(-0.03,1.05), 
          loc = 'upper left',
          columnspacing=0.5, 
          handletextpad = 0.3, 
          labelspacing = 0.3)

#ax.text(12, 8, 'C-rates', color = 'k', horizontalalignment = 'center', verticalalignment = 'center')



ax.set_ylabel('Areal Discharge Capacity \n[mAh/cm$^2$]')
ax.set_xlabel('Thickness [$\mu m$]')

fig.tight_layout()
plt.show()

#%% Penetration depth vs Crate double column figure

Cols = ['#e36414','#0db39e','#748cab','#8a5a44']
markers = ['o','s','d', 'v']
lines = ['-',':','--','']
def stylefun(G):
    return (Cols[G], markers[G], lines[G])

fig, ax= plt.subplots(figsize=(Col2,Col1*AR*1.2))    

Lfun = lambda R, k: np.sqrt(k/R)
Rfun = lambda L, k: k/L**2 

xmin = 1e-1
xmax = 2.1e1
ymin = 0
ymax = 200

ax.set_ylim((ymin, ymax))
ax.set_xlim((xmin, xmax))

xx = np.logspace(np.log10(xmin),np.log10(xmax), 100)

kkcms = np.array([2**i for i in range(-1,8,2)])*1e-8
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
        ax.text(xpos, ypos, "RL$_d^2$ = {}".format(kkcms[i]/1e-8) +  " [$10^{-8}$ cm$^2$/s]", color = 'k', fontsize = 8, horizontalalignment = 'left', verticalalignment = 'top', rotation = -22)


Cc = dfCrit.loc[dfCrit['Batch']=='211202_NMC', 'C-rate(rnd)']
index2 = (Cc > 0.1) & (Cc < 4.8) 
ms = 4
lw = 1.5
for i, (name, group) in enumerate(dfCrit.groupby(by=['Batch'])):
    
    df = group
    
    
    index = ~df['Thickness_lo(um)'].isnull() & ~df['Thickness_hi(um)'].isnull()
    
    if name == '211202_NMC': 
        index = index & index2
        label = 'Current Work, NMC'
    else:
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

ax.set_ylabel('Penetration Depth \nL$_d$ [$\mu m$]')
ax.set_xlabel('C-rate, R [h$^{-1}$]')

ax.set_xscale('log')


handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, 
          ncol = 2, 
          framealpha = 0, 
          fontsize = 8,
          bbox_to_anchor=(-0.01,-0.03), 
          loc = 'lower left',
          columnspacing=0.5, 
          handletextpad = 0.3, 
          labelspacing = 0.3)



fig.tight_layout()
plt.show()

#%% AreaCap vs Cycle



ProgSetting = 2


if ProgSetting == 0:
    Prog = 'Martin_cycles_1'
    NCol = 1
elif ProgSetting == 1:
    Prog = 'Martin_cycles_2'
    NCol = 2
elif ProgSetting == 2:
    Prog = 'Martin_cycles_3'
    NCol = 2

Cycfont = 8

ArMsSetting = 0

if ArMsSetting == 0:
    yCol = 'Avg_DCapacity(mAh/cm2)'
    yLab = 'Discharge Capacity \n[mAh/cm$^2$]'
    ymax = 9
    clabp = 1
    claby = 7
    clabrot = -30
    Leg_kwargs = {'loc': 'upper center', 'bbox_to_anchor' : (0.65,1.02)}

    
elif ArMsSetting == 1:
    yCol = 'Avg_DCapacity(mAh/gAM)'
    yLab = 'Discharge Capacity \n[mAh/g$_{AM}$]'
    ymax = 270
    clabp = 20 
    claby = 240
    clabrot = -5
    NCol = 1
    Leg_kwargs = {'loc': 'lower left', 'bbox_to_anchor' : (-0.02,-0.02)}
    

index = (dfall['Batch'] == '211202_NMC') & (dfall['Cycler_Program'] == Prog)
dfplot = dfall.loc[index, ('SampleID', 'Cycle_ID', 'C-rate(rnd)', 'C-rate(mean)', 'Thickness(um)', 'Avg_DCapacity(mAh/cm2)', 'Avg_DCapacity(mAh/gAM)', 'Wet_Thickness(um)')].copy()

fig, ax= plt.subplots(figsize=(Col1,Col1*AR*1.2))   

plt.ylim((0,ymax))
plt.xlim((0,33)) 

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
    
    return col


def markerfun(i):
    return markers[np.mod(i,len(markers))]



for i, (name, group) in enumerate(dfplot.groupby(by=['SampleID'])):
    

    Thickness = group.loc[:,'Thickness(um)'].unique()[0]
    WT = group.loc[:,'Wet_Thickness(um)'].unique()[0]

    
    group.plot(x = 'Cycle_ID', 
               y=yCol, 
               label = '{}'.format(Thickness), 
               ax=ax, 
               marker = markerfun(i), 
               color = colfun(WT), 
               markersize = 3)
    



handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, 
          ncol = NCol,
          framealpha = 0,
          columnspacing=0.5, 
          handletextpad = 0.3,
          fontsize = 8,
          labelspacing = 0.3,
          **Leg_kwargs)
    
plt.xlabel('Cycle')
plt.ylabel(yLab) 

crts = dfplot['C-rate(mean)'].unique().round(1)
crts = np.append(crts[~np.isnan(crts)], 0.1)    

Cycle_ID_Crate = [1.5, 4, 7, 10, 13, 16, 19, 22, 25, 27.5, 30.5]

for i, (name, group) in enumerate(dfplot.groupby(by=['C-rate(rnd)'])):
    
    
    y = group[yCol].max() + clabp
    if (i==9) | (i==0): 
        ax.text(Cycle_ID_Crate[i], y-clabp/2, str(crts[i]), color = 'k', fontsize = Cycfont, horizontalalignment = 'center')
    else:
        ax.text(Cycle_ID_Crate[i], y, str(crts[i]), color = 'k', fontsize = Cycfont, horizontalalignment = 'center')
    
    if i==0: 
        ax.text(Cycle_ID_Crate[-1], y-clabp/2, str(crts[i]), color = 'k', fontsize = Cycfont, horizontalalignment = 'center')
    
ax.text(8, claby, 'C-rates', color = 'k', fontsize = Cycfont, horizontalalignment = 'center', rotation = clabrot)    
    
fig.tight_layout()
plt.show()
    




#%%

    
    
    
#%%

