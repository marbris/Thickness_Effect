
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

filename='Data/Data_220621_LFP/minipunch_data_2.csv'
df = pd.read_csv(filename)


diameter = 0.5 #cm
Area = (diameter/2)**2*np.pi #cm2

Al_thickness = 22e-4 #cm
Al_density = 2.7 #g/cm3
IPA_density = 0.785 #g/cm3
Air_density = 1.225e-3 #g/cm3

ECC_mass_Air = df['Mass_Air [mg]'].to_numpy(dtype=float) #mg
ECC_mass_IPA = df['Mass_IPA [mg]'].to_numpy(dtype=float) #mg
ECC_Thickness = df['Thickness [um]'].to_numpy(dtype=float) #um

#%%

fig, ax = plt.subplots(figsize=(7,3))

plt.plot(ECC_Thickness,ECC_mass_Air,
        color = 'r', linestyle = '', marker = '.', label = 'Weighed in Air')

plt.plot(ECC_Thickness, ECC_mass_IPA,
        color = 'b', linestyle = '', marker = '.', label = 'Weighed in IPA')

ax.set_ylim((0,6))
ax.set_xlim((0,240))

ax.set_ylabel('E+CC Mass [mg]')
ax.set_xlabel('E+CC Thickness [um]')

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, 
          framealpha = 0.0, 
          fontsize = 10,
          #bbox_to_anchor=(1.00,0.96), 
          loc = 'upper left')

fig.tight_layout()

#%%

fig, ax = plt.subplots(figsize=(7,3))

ECC_Density = (ECC_mass_IPA*Air_density - ECC_mass_Air*IPA_density)/(ECC_mass_IPA - ECC_mass_Air)

plt.plot(ECC_Thickness,ECC_Density,
        color = 'g', linestyle = '', marker = '.')

ax.set_ylim((0,6))
ax.set_xlim((0,240))

ax.set_ylabel('E+CC Density [g/cm$^2$]')
ax.set_xlabel('E+CC Thickness [um]')

# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels, 
#           framealpha = 0.0, 
#           fontsize = 10,
#           #bbox_to_anchor=(1.00,0.96), 
#           loc = 'upper left')

fig.tight_layout()

#%%

fig, ax = plt.subplots(figsize=(7,3))


ECC_Volume = ECC_mass_Air*1e-3/(ECC_Density-Air_density)*1e3 #mm3

E_volume = ECC_Volume - Al_thickness*Area*1e3 #mm3

E_outer_volume = (ECC_Thickness*1e-4-Al_thickness)*Area*1e3 #mm3


E_porosity = 1-E_volume/(E_outer_volume)

plt.plot(ECC_Thickness,E_porosity,
        color = 'r', linestyle = '', marker = '.')

#ax.set_ylim((0,6))
ax.set_xlim((0,240))

ax.set_ylabel('E porosity')
ax.set_xlabel('E+CC Thickness [um]')

fig.tight_layout()

# %%
