#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 15:45:49 2021

@author: martin
"""
#%%

import Cycler
from Cycler import BTS_csv2json, MITS_2json
import pandas as pd

import os
import re


def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r




#%%

#datafolder = r'Data/Data_220203_NMC/Martin_0203_2_2022_03_04_172442'
#datafolder = "Data/Data_220203_NMC/raw"
datafolder = 'Data/Data_211202_NMC/CoinCell_RAW_211202_NMC'
r = list_files(datafolder)



regexp=".*\.CSV$"

for file in r:
    if re.search(regexp, file):
        MITS_2json(file, CSV=True)
        print("Finished: {}".format(file))
                



#%%


r = list_files(r'Data/')


regexp=r"CoinCell_.*\.csv$"

for file in r:
    if re.search(regexp, file):
        BTS_csv2json(file)
        print("Finished: {}".format(file))

#%%

datafolder = r'Data/Data_220203_NMC/Martin_0203_2_2022_03_04_172442'
r = list_files(datafolder)


regexp=r"*\.xlsx$"


for file in r:
    sampleID = re.findall(regexp, file)
    if sampleID:
        folder =os.path.split(file)[0]
        outfile = os.path.join(datafolder,sampleID[0] + '.json')
        
        #MITS_csv2json(file, outfile=outfile)
        print("Finished: {}".format(sampleID[0]))


#%%


r = list_files(r'Data/Data_220209_NMC/')


regexp=r"CoinCell_.*\.CSV$"

for file in r:
    if re.search(regexp, file):
        MITS_2json(file, CSV=True)
        print("Finished: {}".format(file))
        

#%%

MITS_2json('CoinCell_14_220209_NMC.CSV', outfile='CoinCell_14_220209_NMC_PrettyJSON_MITS.json', Pretty_JSON=True, CSV=True)

#%%



df = Cycler.get_SampleData('220209_NMC', '14', 'Data/')

Allkeys = list(df['Cycles'][0]['Steps'][0])

CycleDataKeys = []
for key in Allkeys:
    if not isinstance(df['Cycles'][0]['Steps'][0][key], list):
        CycleDataKeys.append(key)






#%%

filename = 'Data/Data_220203_NMC/Martin_0203_2_2022_03_04_172442/Martin_0203_2_Channel_30_Wb_1.xlsx'

D = pd.read_excel(filename, sheet_name = 1)


#%%


