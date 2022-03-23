
#%%
import Cycler
import main
from matplotlib import pyplot as plt
import numpy as np


SMALL_SIZE = 8
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


#%% 



Cols = ['#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6', '#ee9b00', '#ca6702', '#bb3e03', '#ae2012', '#9b2226']

def colfun(i):
    return Cols[np.mod(i-1,len(Cols))]


Batch = '211202_NMC'
Sample = '16'

df_pnt = Cycler.get_PointData(Batch, Sample, Properties = ['Cycle_Index', 'Step_Index', 'Voltage(V)', 'Charge_Capacity(Ah)', 'Discharge_Capacity(Ah)','Current(A)'])

OPdf, df_dQdV = Cycler.get_overpotential(Batch, Sample, Curve_Volt=0.08)

dfall, dfCrit = main.dfall_dfCrit()

#df = df_pnt.loc[df_pnt['Cycle_Index']==2,:]

#xx = np.array([])
cycles = np.array([3,6,9])
#cycles = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,31]

#%%





fig= plt.figure(figsize=(Col2,Col1*AR*1.2))
ax= fig.add_axes([0.08,0.2,0.9,0.75])

ax.set_xlim([3.6,4.4])
ax.set_ylim([-4e1,2e1])

ax.set_xlabel('Voltage (V)')
ax.set_ylabel('dQ/dV (mAh/V)')

def barfun(Cyc):
    return -25-Cyc
    


i=0

for (Cyc, Step), df in df_pnt.groupby(['Cycle_Index','Step_Index']):

    TotCap = df.loc[:, 'Current(A)'].sum() 
    if TotCap > 0:
        mode = "Charge"
    elif TotCap < 0:
        mode = "Discharge"
    else:
        mode = "Rest"  
    
    
    if (Cyc in cycles) & (mode != "Rest"):

        V = df['Voltage(V)'].to_numpy(dtype=float)
        
        Peak_dc = []
        
        for MODE in ['Discharge', 'Charge']:
            #print('C: {}, S: {}'.format(Cyc,Step))
            Q = df['{}_Capacity(Ah)'.format(MODE)].to_numpy(dtype=float)
            
        
            dV0_index = np.diff(V)==0
            # dV0_index is one element shorter than V and Q. so I'm adding a False value at the end. Then the True values will correspond to the FIRST of the pair V[i]-V[i+1]=0 
            dV0_index1 = np.append(dV0_index,False) #first
            dV0_index2 = np.append(False, dV0_index) #second
            
            #calculating the average Q and placing it in the first of the pairs
            Q[dV0_index1] = (Q[dV0_index1] + Q[dV0_index2])/2
            #here im removing the second element of the pair, both from Q and V.
            Q = np.delete(Q, dV0_index2)
            Vtemp = np.delete(V, dV0_index2)
            
            dQdV = np.gradient(Q,Vtemp)
            
            
            #removing points that are almost zero or nan
            nanzero_mask = (abs(dQdV)>1e-5) & (~np.isnan(dQdV))
            dQdV = dQdV[nanzero_mask]
            Vtemp = Vtemp[nanzero_mask]
            
            
            plt_data_kwargs = {  'marker' : '.', 
                            'linestyle' : '', 
                            'markersize' : 1.5, 
                            'linewidth' : 1, 
                            'color' : colfun(Cyc)}
            
            
            if MODE == 'Charge':
                #some points are negative and look bad next to the discharge curve.
                extra_index = dQdV>2e-3
                
                Mode_kwargs = {}
                
            else:
                extra_index = [True] * len(dQdV)
                
                #Im getting the c-rate for the discharge curves
                index = (dfall['Batch'] == Batch) & (dfall['SampleID'] == Sample) & (dfall['Cycle_ID'] == Cyc)
                crate = dfall.loc[index,'C-rate(1/h)'].values[0]
                
                Mode_kwargs = {}
            
            
            dQdV = dQdV[extra_index]
            Vtemp = Vtemp[extra_index]
            
            
            ax.plot(Vtemp,dQdV*1e3,**plt_data_kwargs, **Mode_kwargs)
            
            #i +=1
            #print(i)
            
            plt_smooth_kwargs = {'color' : colfun(Cyc),
                        'linewidth' : 3, 
                        'alpha' : 0.25, 
                        'marker' : ''}
            index = (df_dQdV['Cycle_ID'].values == Cyc) & (df_dQdV['mode'].values == MODE)
            X = df_dQdV.loc[index,'Voltage(V)'].to_numpy(dtype=float)
            Y = df_dQdV.loc[index,'dQdV_smooth(Ah/V)'].to_numpy(dtype=float) * 1e3 #mAh/V
            ax.plot(X,Y,**plt_smooth_kwargs)
            
            peak_V = OPdf.loc[OPdf['Cycle_ID']==Cyc, '{}_peak(V)'.format(MODE)].values[0]
            peak_dQdV = OPdf.loc[OPdf['Cycle_ID']==Cyc, '{}_peak_dQdV(Ah/V)'.format(MODE)].values[0]*1e3
            plt.vlines(peak_V, barfun(Cyc), peak_dQdV,color = colfun(Cyc))
            
            Peak_dc.append(peak_V)
            
        
        index = (dfall['Batch'] == Batch) & (dfall['SampleID'] == Sample) & (dfall['Cycle_ID'] == Cyc)
        crate = dfall.loc[index,'C-rate(1/h)'].values[0]
        
        
        plt.hlines(barfun(Cyc), min(Peak_dc), max(Peak_dc), color = colfun(Cyc))
        OP = OPdf.loc[OPdf['Cycle_ID']==Cyc, 'Overpotential(V)'].values[0]
        
        plt.text(min(Peak_dc)+0.005,barfun(Cyc)+2, 'C-rate: {:.1f} h$^{{-1}}$ \n $\eta_0$ = {:.0f} mV'.format(crate, OP*1e3), fontsize = 7)
            
    

        
        # plt_kwargs = {  'ax' : ax,
        #                 'color' : colfun(Cyc),
        #                 'linestyle' : '', 
        #                 'alpha' : 0.5, 
        #                 'marker' : 'o'}
        
        # index = cycindex & (df_dQdV['mode'].values == 'Charge')
        # df_dQdV.loc[index,:].plot(x='Voltage(V)', 
        #                         y = 'dQdV(Ah/V)', 
        #                         **plt_kwargs)
        
        # index = cycindex & (df_dQdV['mode'].values == 'Discharge')
        # df_dQdV.loc[index,:].plot(x='Voltage(V)', 
        #                         y = 'dQdV(Ah/V)', 
        #                         **plt_kwargs)



plt.text(4.2,15, 'C-rate: C/3', fontsize = 7)

#dummyplots for the legend
plt_kwargs = plt_data_kwargs
plt_kwargs['color'] = 'k'

ax.plot([np.nan, np.nan],[np.nan, np.nan], **plt_kwargs, label = 'data points')

plt_kwargs = plt_smooth_kwargs
plt_kwargs['color'] = 'k'

ax.plot([np.nan, np.nan],[np.nan, np.nan], **plt_kwargs, label = 'smoothed')
 
        
handles, labels = ax.get_legend_handles_labels()



ax.legend(handles[:2], labels[:2],
          loc='upper left', 
          framealpha = 0,
          columnspacing=0.5, 
          handletextpad = 0.3,
          labelspacing = 0.3,
          ncol = 1)   


fig.tight_layout() 
    
#%%



fig2= plt.figure(figsize=(Col2,Col1*AR*1.2))
ax2= fig2.add_axes([0.15,0.2,0.8,0.75])

ax2.set_xlim([2.8,4.4])
ax2.set_ylim([-4e1,4e1])
ax2.set_xlabel('Voltage (V)')
ax2.set_ylabel('dQ/dV (mAh/V)')
ax2_kwargs = {'marker' : '.',
             'linestyle' : '',
             'markersize' : 2}


cycles = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,31]

for (Cyc, Step), df in df_pnt.groupby(['Cycle_Index','Step_Index']):

    TotCap = df.loc[:, 'Current(A)'].sum() 
    if TotCap > 0:
        mode = "Charge"
    elif TotCap < 0:
        mode = "Discharge"
    else:
        mode = "Rest"  
    
    
    if (Cyc in cycles) & (mode != "Rest"):

        V = df['Voltage(V)'].to_numpy(dtype=float)
        
        for MODE in ['Discharge', 'Charge']:
            #print('C: {}, S: {}'.format(Cyc,Step))
            Q = df['{}_Capacity(Ah)'.format(MODE)].to_numpy(dtype=float)
            
        
            dV0_index = np.diff(V)==0
            # dV0_index is one element shorter than V and Q. so I'm adding a False value at the end. Then the True values will correspond to the FIRST of the pair V[i]-V[i+1]=0 
            dV0_index1 = np.append(dV0_index,False) #first
            dV0_index2 = np.append(False, dV0_index) #second
            
            #calculating the average Q and placing it in the first of the pairs
            Q[dV0_index1] = (Q[dV0_index1] + Q[dV0_index2])/2
            #here im removing the second element of the pair, both from Q and V.
            Q = np.delete(Q, dV0_index2)
            Vtemp = np.delete(V, dV0_index2)
            
            dQdV = np.gradient(Q,Vtemp)
            
            
            #removing points that are almost zero or nan
            nanzero_mask = (abs(dQdV)>1e-5) & (~np.isnan(dQdV))
            dQdV = dQdV[nanzero_mask]
            Vtemp = Vtemp[nanzero_mask]
            
            ax2.plot(Vtemp, dQdV*1e3, color=colfun(Cyc), **ax2_kwargs)