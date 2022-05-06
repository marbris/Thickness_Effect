





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




RFile='Data/McMullin/RIon_WT_220428_NMC.csv'
df_RW = pd.read_csv(RFile)

Area = (1.27/2)**2*np.pi #cm2
Cond = 3300*1e-6 #S/cm


sum_d = df_RW['Thickness_Sum[um]'].to_numpy(dtype=float)*1e-4 #cm
Rion = df_RW['Rion[Ohm]'].to_numpy(dtype=float) #Ohm
WT = df_RW['Wet_Thickness[um]']

Rion_err = df_RW['Rion_err[Ohm]'].to_numpy(dtype=float) #Ohm
Cond_err = Cond*0.01 #S/cm
Area_err = Area*0.05 #cm2
sum_d_err = sum_d*0.01 #cm

NM = Rion*Area*Cond/sum_d
NM_err = NM*np.sqrt((Rion_err/Rion)**2 + (Cond_err/Cond)**2 + (Area_err/Area)**2 + (sum_d_err/sum_d)**2)

invNM = 1./NM
invNM_err = invNM*np.sqrt((Rion_err/Rion)**2 + (Cond_err/Cond)**2 + (Area_err/Area)**2 + (sum_d_err/sum_d)**2)

#%%

fig, ax = plt.subplots(figsize=(7,3))


plt.errorbar(sum_d/2*1e4,invNM, xerr = sum_d_err/2*1e4, yerr = invNM_err, marker = 'o', linestyle ='-')

ax.set_ylabel(r'$\frac{\epsilon}{\tau}$', rotation = True, fontsize = 20)
ax.set_xlabel(r'Cathode Thickness, [um]')
txt = r'$D_{eff} = \frac{\epsilon}{\tau}D_{bulk}$'
#ax.text(120,40,txt, fontsize = 20)
ax.text(60,0.06,txt, fontsize = 20)

fig.tight_layout()

#%%

fig, ax = plt.subplots(figsize=(7,3))

porosity = 0.25
porosity_err = 0.05

tortuosity = NM*porosity
tortuosity_err = tortuosity*np.sqrt((NM_err/NM)**2 + (porosity_err/porosity)**2)

plt.errorbar(sum_d/2*1e4,tortuosity, yerr = tortuosity_err, marker = 'o', linestyle ='-')

ax.set_ylabel(r'Tortuosity, $\tau$')
ax.set_xlabel(r'Cathode Thickness, [um]')
txt = r'$D_{eff} = \frac{\epsilon}{\tau}D_{bulk}$'
#ax.text(120,40,txt, fontsize = 20)
#ax.text(60,0.06,txt, fontsize = 20)

fig.tight_layout()

#%%

ax.set_ylim((0,20))
ax.set_xlim((0,60))

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, 
          framealpha = 0.0, 
          fontsize = 10,
          #bbox_to_anchor=(1.00,0.96), 
          loc = 'upper left',
          columnspacing=0.5, 
          handletextpad = 0.3, 
          labelspacing = 0.3)


fig.tight_layout()
# %%


