

#%%
import Cycler
import main
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import simplejson as json


SampleList = {
                        #'220203_NMC': ['10', '11', '15', '16', '24', '25'],
                        #'220203_NMC': ['03', '04', '05', '06', '09', '17', '18', '21', '22', '23'],
                        '220203_NMC': ['03', '06', '18', '21', '22', '23'],
                        '211202_NMC': ['02', '03', '05', '06', '07', '08', '09', '12', '13', '15', '16', '17', '18', '19']
                    }

#%%
#3,4,6, 18
#21,22,23
#Batch = '220203_NMC'
#Sample = '18'

Batch = '211202_NMC'
Sample = '19'

df = Cycler.get_overpotential(Batch, Sample, Plot=1)

#%%

SampleList = {'220203_NMC': ['03', '04', '06', '18', '21', '22', '23']}

dfOP = Cycler.OPSampleList(SampleList)

#%%

fig , ax = plt.subplots(figsize=(10,6))

for name , group in dfOP.groupby(by='Sample'):
    group.plot(x='Cycle', y='Overpotential(V)', ax=ax, marker = 'o', label = name)

#%%

d = {'col1' : [1,1,1,1,2,2,2,2,3,3,3,3], 'col2': [4,4,4,5,5,5,6,6,6,7,7,7]}
dft = pd.DataFrame(data=d)

groups = dft.groupby(by='col1')




# %%
with open('Data/Supplemental/Cycler_Prog.json') as file:
        CyclerProg=json.load(file)
        
        
# %%


#dfall, dfCrit = main.dfall_dfCrit(read_json = False, write_json = True)



