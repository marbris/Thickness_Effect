

#%%
import Cycler
import main
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import simplejson as json

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

dfOPCrit = Cycler.get_OPCrit(dfall)
        


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



