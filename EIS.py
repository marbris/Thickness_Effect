
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as scipy_optimize
import main
import os
import re
import Cycler

def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r

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

plt.rc('figure', dpi = 150)

Col1= 8.9/2.54 # 8.9cm single column figure width in Nature
Col2= 18.3/2.54 # 18.3cm double column figure width in Nature

AR = 1/1.618




EISfilepath = 'Data/McMullin/EIS Symmetric cells'

r = list_files(EISfilepath)
r.sort()


headers = ["nothin","Pt[#]", "Time[s]", "Freq[Hz]", "Zreal[ohm]", "Zimag[ohm]", "Zsig[V]", "Zmod[ohm]", "Zphz[deg]", "Idc[A]", "Vdc[V]", "IERange[#]"]


#%%
def Zfun(w, R, Q, a):
    ZS = 1/(Q*(1j*w)**a)
    Z = np.sqrt(R*ZS)/np.tanh(np.sqrt(R/ZS))
    return Z

R = 42.7
Q = 0.0072
a = 0.87

x = np.logspace(-1,5,1000)

y = Zfun(x,R,Q,a)




#fig, ax = plt.subplots(figsize=(13,6))

#plt.plot(np.real(y),-np.imag(y),marker = '.', linestyle = '')


#%%


regexp=".*_Data\.DTA$"

Cols2 = ['#e76f51', '#f4a261', '#e9c46a', '#2a9d8f', '#264653']
def colfun(i):
    return Cols2[np.mod(i,len(Cols2))]

WT = {  '0405': 200,
        '1417': 300,
        '2024': 400,
        '2527': 500,
        '2930': 600}


#sampleinfoFile='Data/Data_220428_NMC/inputs.csv'
#df_SI = pd.read_csv(sampleinfoFile)

LowZ = False

if LowZ:
    figsize = (7,3)
    ylim = (0,20)
    xlim = (0,60)
    data_plt_kwargs = { 'markersize' : 5}
    
else:
    figsize = (3,7)
    ylim = (0,1300)
    xlim = (0,170)
    data_plt_kwargs = { 'markersize' : 5}



#fig, ax = plt.subplots(figsize=(7,3))
fig, ax = plt.subplots(figsize=figsize)

sampleinfoFile='Data/McMullin/RIon_WT_220428_NMC.csv'
df_SI = pd.read_csv(sampleinfoFile)

Rion = np.array([])
Rhfr = np.array([])
for i, file in enumerate(r):
    if re.search(regexp, file):
        
        samples = file[-13:-9]
        
        D = pd.read_csv(file, sep = '\t',encoding_errors='ignore', skiprows = 54,header = 0, names = headers)
        
        Freq = D['Freq[Hz]'].to_numpy(dtype=float)
        Zreal = D['Zreal[ohm]'].to_numpy(dtype=float)
        Zimag = D['Zimag[ohm]'].to_numpy(dtype=float)
        #Z = Zreal + 1j*Zimag
        
        #p0 = [50, 1e-2, 0.8]
        #popt, pcov = scipy_optimize.curve_fit(EISfun, Freq, Z, p0=p0)
        
        #index = (dfall['Batch'] == '211202_NMC') & (dfall['Wet_Thickness(um)'] == WT[samples])
        
        #index = df_SI['DoctorBlade[um]'] ==  WT[samples]
        #List_Thickness = df_SI.loc[index,'thickness[um]'].tolist()
        
        #index = df_SI['DoctorBlade[um]'] ==  WT[samples]
        #List_Thickness = df_SI.loc[index,'thickness[um]'].tolist()
        
        #List_Thickness = dfall.loc[index, 'Thickness(um)'].unique()
        index = df_SI['Wet_Thickness[um]'] == WT[samples]
        thickness = df_SI.loc[index, 'Thickness_Sum[um]'].to_numpy(dtype=float)/2
        
        plt.plot(Zreal, -Zimag, marker = '.', linestyle = '', color = colfun(i), label = '{:.0f} $\mu$m'.format(thickness[0]), **data_plt_kwargs)
        
        Index = Freq < 0.5
        #plt.plot(Zreal[Index], -Zimag[Index], marker = 'o', linestyle = '', color = colfun(i))
        
        p_ion, pcov = scipy_optimize.curve_fit( lambda x, k, m: k*x + m,Zreal[Index], -Zimag[Index])
        
        perr_ion = np.sqrt(np.diag(pcov))
    
        
        #p_ion = np.polyfit(Zreal[Index], -Zimag[Index], 1)
        
        x0_ion = -p_ion[1]/p_ion[0]
        
        x0_ion_err = np.sqrt((perr_ion[0]*x0_ion/p_ion[0])**2 + (perr_ion[1]*x0_ion/p_ion[1])**2)
        
        
        
        X=np.array([x0_ion, max(Zreal)])
        Y=np.polyval(p_ion, X)
        plt.plot(X,Y,linestyle = '-', color = colfun(i) )
        
        plt.errorbar(x0_ion, 0.1, xerr = x0_ion_err, marker = 'o', linestyle = '-', color = colfun(i) , zorder = 10, capsize = 5)
        
        
        
        
        Index = Freq > 5e4
        #plt.plot(Zreal[Index], -Zimag[Index], marker = 'o', linestyle = '', color = colfun(i))
        
        p_hfr = np.polyfit(Zreal[Index], -Zimag[Index], 1)
        
        R_hfr = -p_hfr[1]/p_hfr[0]
        
        X=np.array([R_hfr, 5])
        Y=np.polyval(p_hfr, X)
        plt.plot(X,Y,linestyle = '-', color = colfun(i))
        
        R_ion = (x0_ion - R_hfr)*3
    
        R_ion_err = x0_ion_err*3
        
        Rion = np.append(Rion,R_ion)
        Rhfr = np.append(Rhfr,R_hfr)
        
        
        #print(f'{R_ion},{R_hfr},{R_ion_err}')
        print(f'{R_ion_err}')
        #D.plot(x = 'Zreal[ohm]', y = 'Zimag[ohm]', marker = '.', linestyle = '', ax=ax, label = file)

    
ax.set_ylim(ylim)
ax.set_xlim(xlim)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, 
          framealpha = 0.0, 
          fontsize = 10,
          #bbox_to_anchor=(1.00,0.96), 
          loc = 'upper left',
          columnspacing=0.5, 
          handletextpad = 0.3, 
          labelspacing = 0.3)

ax.set_xlabel('Re(Z) [$\Omega$]')
ax.set_ylabel('-Im(Z) [$\Omega$]')

fig.tight_layout()
# %%


