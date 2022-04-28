#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:22:52 2022

@author: martin
"""

import Cycler
import pandas as pd
import simplejson as json


def dfall_dfCrit(read_json = True, write_json = False, SampleList = {}):

    """
    gets dfall and dfCapCrit. 
    
    if read_json=True (default), the dataframes are imported from file. Otherwise they're calculated from data.
    
    if write_json=True (default: false), the calculated dataframes are written to .json files. these are overritten. This also only happens if the dataframes are calculated, ie read_json=False.

    Returns:
       dfall, dfCapCrit Pandas dataframes: 
    """
    
    if not read_json:

        if SampleList == {}:
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
        
        dfCapCrit = Cycler.get_CapCrit(dfall)
        
        dfOPCrit = Cycler.get_OPCrit(dfall)

        
        if write_json:
            dfall.to_json("dfall.json")
            dfCapCrit.to_json("dfCapCrit.json")
            dfOPCrit.to_json("dfOPCrit.json")
        
    else:
        
        
        with open("dfall.json") as file:
            dfall_dict=json.load(file)
            dfall = pd.DataFrame.from_dict(dfall_dict)
        
        with open("dfCapCrit.json") as file:
            dfCapCrit_dict=json.load(file)
            dfCapCrit = pd.DataFrame.from_dict(dfCapCrit_dict)
            
        with open("dfOPCrit.json") as file:
            dfOPCrit_dict=json.load(file)
            dfOPCrit = pd.DataFrame.from_dict(dfOPCrit_dict)
        

    
    return dfall, dfCapCrit, dfOPCrit




