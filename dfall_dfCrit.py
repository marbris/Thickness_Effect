#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:22:52 2022

@author: martin
"""

import Cycler
import pandas as pd
import simplejson as json


def dfall_dfCrit(read_json = True, write_json = False):

    """
    gets dfall and dfcrit. 
    
    if read_json=True (default), the dataframes are imported from file. Otherwise they're calculated from data.
    
    if write_json=True (default: false), the calculated dataframes are written to .json files. these are overritten. This also only happens if the dataframes are calculated, ie read_json=False.

    Returns:
       dfall, dfCrit Pandas dataframes: 
    """
    
    if not read_json:

        SampleList = {
                        #'220203_NMC': ['10', '11', '15', '16', '24', '25'],
                        #'220203_NMC': ['03', '04', '05', '06', '09', '17', '18', '21', '22', '23'],
                        '220203_NMC': ['03', '04', '06', '18', '21', '22', '23'],
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
        dfall = pd.merge(OPall, dfall, on=['Cycle_ID', 'SampleID', 'Batch'], how='outer')
        
        # creating c-rate groups
        ProgDict = {
                    'Martin_cycles_1': [0.1, 0.5, 0.8, 1.2, 1.8, 2.7, 4.2, 6.4, 9.8, 15.1],
                    'Martin_cycles_2': [0.1, 0.6, 0.9, 1.3, 2.1, 3.2, 4.8, 7.4, 11.3, 17.4],
                    'Martin_cycles_3': [0.1, 0.7, 1, 1.6, 2.4, 3.6, 5.6, 8.5, 13.1, 20],
                    '01C_1C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                    '01cC_01dC_1dC': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                    'H. Zheng (2012), NMC': [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50],
                    'H. Zheng (2012), LFP': [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 100, 200, 300],
                    'D.Y.W. Yu (2006), LFP': [0.1, 0.2, 0.5, 1, 2, 5, 8, 12, 18, 25]
                    }
        
        #here i'm groupding the c-rates within each Batch.
        dfall = dfall.groupby('Batch').apply(Cycler.CRate_groups, ProgDict=ProgDict)
        
        
        dfCrit = Cycler.get_CritRL(dfall)
        
        if write_json:
            dfall.to_json("dfall.json")
            dfCrit.to_json("dfCrit.json")
        
    else:
        
        
        with open("dfall.json") as file:
            dfall_dict=json.load(file)
            dfall = pd.DataFrame.from_dict(dfall_dict)
        
        with open("dfCrit.json") as file:
            dfCrit_dict=json.load(file)
            dfCrit = pd.DataFrame.from_dict(dfCrit_dict)
        

    
    return dfall, dfCrit




