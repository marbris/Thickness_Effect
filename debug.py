

#%%
import Cycler
import main
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import simplejson as json

#%

SampleList = {
                        #'220203_NMC': ['10', '11', '15', '16', '24', '25'],
                        #'220203_NMC': ['03', '04', '05', '06', '09', '17', '18', '21', '22', '23'],
                        #'220203_NMC': ['03', '06', '18', '21', '22', '23'],
                        #'211202_NMC': ['02', '03', '05', '06', '07', '08', '09', '12', '13', '15', '16', '17', '18', '19'],
                        '220303_NMC' : ['01', '02', '03', '04', '06', '07'],
                        '220317_NMC' : ['01', '02', '03', '04', '05', '07']
                    }
        
#collecting batch cycle data 
#dfall = Cycler.get_BatchCycles(SampleList)

#dfall = dfall.groupby('Batch').apply(Cycler.CRate_groups)

dfall, dfCrit, dfOPCrit = main.dfall_dfCrit(read_json = False, write_json = True)#, SampleList=SampleList)
#%%




fig, ax= plt.subplots(figsize=(13,8))   


Cols = ['#e76f51', '#f4a261', '#e9c46a', '#2a9d8f', '#264653']

def colfun(i):
    colorlist = Cols
    return colorlist[np.mod(i,len(colorlist))]

def marksizefun(t):
    return (t - 220)/(363-220)*4 + 3

#batchlist = ['220303_NMC', '220317_NMC']
batchlist = [ '220317_NMC']
index = dfall['Batch'].isin(batchlist) & (dfall['Cycle']<30)

for i, ((crate, batch), group) in enumerate(dfall.loc[index,:].groupby(by = ['Avg_C-rate(1/h)', 'Batch'])):
    
    
    group.sort_values(by = ['Thickness(um)', 'Cycle'], inplace = True)
    
    
    group.plot(x='Thickness(um)', y = 'Discharge_Capacity(mAh/cm2)', color = colfun(i), ax=ax, marker = 'o', label = "C = {:.2f}".format(crate))

ax.set_xscale('log')

ax.set_xlabel('Thickness(um)')
ax.set_ylabel('Discharge_Capacity(mAh/cm2)')

fig.tight_layout()
plt.show()

#%%


SampleList = {
                        #'220203_NMC': ['10', '11', '15', '16', '24', '25'],
                        #'220203_NMC': ['03', '04', '05', '06', '09', '17', '18', '21', '22', '23'],
                        '220203_NMC': ['03', '06', '18', '21', '22', '23'],
                        '211202_NMC': ['02', '03', '05', '06', '07', '08', '09', '12', '13', '15', '16', '17', '18', '19']
                    }


with open("dfall.json") as file:
            dfall_dict=json.load(file)
            dfall = pd.DataFrame.from_dict(dfall_dict)

#%%


    
    
    

#%%

dfall, dfCrit, dfOPCrit = main.dfall_dfCrit(read_json = False, write_json = True)

#%%
#3,4,6, 18
#21,22,23
#Batch = '220203_NMC'
#Sample = '18'

Batch = '220203_NMC'
Sample = '03'

SampleList = {
                        #'220203_NMC': ['10', '11', '15', '16', '24', '25'],
                        #'220203_NMC': ['03', '04', '05', '06', '09', '17', '18', '21', '22', '23'],
                        '220203_NMC': ['03', '06', '18', '21', '22', '23'],
                        '211202_NMC': ['02', '03', '05', '06', '07', '08', '09', '12', '13', '15', '16', '17', '18', '19']
                    }

#collecting batch cycle data 
dfall = Cycler.get_BatchCycles(SampleList)

#adding HZheng and DYW Yu
dfnew = Cycler.init_OtherData()

dfall = pd.concat([dfall, dfnew], ignore_index = True)


#here im calculating the peak potentials and overpotentials for each cycle
#it throws a bunch of warnings. thats no problem.
OPall = Cycler.OPSampleList(SampleList)

#merging the overpotential data with dfall
dfall = pd.merge(OPall, dfall, on=['Cycle', 'Sample', 'Batch'], how='outer')


#here i'm groupding the c-rates within each Batch.
dfall = dfall.groupby('Batch').apply(Cycler.CRate_groups)

#%%

#dfall.to_json("dfall.json")


#%%

#dfCapCrit = Cycler.get_CapCrit(dfall)


# %%


#dfall, dfCrit = main.dfall_dfCrit(read_json = False, write_json = True)



