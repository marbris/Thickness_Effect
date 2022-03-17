#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 13:52:27 2021

@author: Martin Brischetto

"""

import pandas as pd
import simplejson as json
import os
import re
from numpy import pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def BTS_csv2json(filename, outfile='', Output_JSON_file=True, Pretty_JSON=False):

    """
    A function to convert the csv data extracted from Neware BTS into json format. It both creates a json-file and outputs a dict object.

    BTS_csv2json(filename, outfile='', Output_JSON_file=True, Pretty_JSON=False)

    filename: the filename of the csv file
    outfile: the filename of the output json file, default '' will take filename and replace .csv with .json.
    Output_JSON_file: if True (default) a json-file will be created, otherwise only the dict will be output.
    Pretty_JSON: if False (default) the json-file will be compacted and not very readable. 
                 if True, the json-file will include indents to make it readable, but will substantially increase the size. May be innappropriate for large data.
    
    
    Below is output format. The numbers are examples, [...] means that the is an array of values there.
    The structure is made up of Cycles, each of which with properties and several steps. 
    Each step also has several properties, including the arrays of data making up the bulk of the file.
        
    {
        "Cycles":
        [
            {
                "Cycle ID": 1,
                "Cap_Chg(mAh)": 6.3758,
                "Cap_DChg(mAh)": 5.7762,
                "Specific Capacity-Chg(mAh/g)": 238.1688,
                "Specific Capacity-Dchg(mAh/g)": 215.77,
                "Chg/DChg Efficiency(%)": 90.595,
                "Engy_Chg(mWh)": 25.2346,
                "Engy_DChg(mWh)": 22.2777,
                "REngy_Chg(mWh/g)": 942.6448,
                "REngy_Dchg(mWh/g)": 832.1885,
                "CC_Chg_Ratio(%)": 100.0,
                "CC_Chg_Cap(mAh)": 6.3758,
                "Plat_Cap(mAh)": 4.9421,
                "Plat_Capacity Density(mAh/g)": 184.6133,
                "Plat_Cap_Efficiency(%)": 85.56,
                "Plat_Time(h:min:s.ms)": "9:15:30.000",
                "Capacity_Chg(F)": 28.03225,
                "Capacity_DChg(F)": 13.70834,
                "IR(O)": 0.0,
                "Mid_value Voltage(V)": 3.8423,
                "Discharge Fading Ratio(%)": 100.0,
                "Charge Time(h:min:s.ms)": "11:56:34.000",
                "Discharge Time(h:min:s.ms)": "10:49:15.000",
                "Charge IR(O)": 239.228,
                "Discharge IR(O)": 34.283,
                "End Temperature(C)": 0.0,
                "Net Cap_DChg(mAh)": -0.5996,
                "Net Engy_DChg(mWh)": -2.9569,
                "Energy Efficiency(%)": 88.282,
                "Steps":
                [
                    {
                        "Step ID": 1,
                        "Step Name": "Rest",
                        "Step Time(h:min:s.ms)": "5:00:00.000",
                        "Step Capacity(mAh)": 0.0,
                        "Step Capacity Density(mAh/g)": 0.0,
                        "Step Energy(mWh)": 0.0,
                        "CmpEngergy(mWh/g)": 0.0,
                        "Capacitance(F)": 0.0,
                        "Start Voltage(V)": 3.4582,
                        "End Voltage(V)": 3.4536,
                        "Start Temperature(C)": 0.0,
                        "End Temperature(C)": 0.0,
                        "End Capacity(mAh)": 0.0,
                        "Charge Mid-Vol(V)": 0.0,
                        "Discharge Mid-Vol(V)": 0.0,
                        "DCIR(O)": "-",
                        "ChargeCap(mAh)": 0.0,
                        "DischargeCap(mAh)": 0.0,
                        "Step Engy_Chg(mWh)": 0.0,
                        "Step Engy_DChg(mWh)": 0.0,
                        "Net DischargeCap(mAh)": 0.0,
                        "Net Engy_DChg(mWh)": 0.0,
                        "Raw Step ID": 1,
                        "Record ID": [...]
                        "Time(h:min:s.ms)": [...]
                        "Voltage(V)": [...]
                        "Current(mA)": [...]
                        "Temperature(C)": [...]
                        "CmpEng(mWh/g)": [...]
                        "Realtime": [...]
                        "Min-T(C)": [...]
                        "Max-T(C)": [...]
                        "Avg-T(C)": [...]
                        "Power(mW)": [...]
                        "Capacitance_Chg(mAh)": [...]
                        "Capacitance_DChg(mAh)": [...]
                        "dQ/dV(mAh/V)": [...]
                        "dQm/dV(mAh/V.g)": [...]
                        "Contact IR(O)": [...]
                    },
                    {
                        "Step ID": 2,
                        ...
                    },
                    {
                        "Step ID": 3,
                        ...
                    }
                    ...
                ]
            },
            {
                "Cycle ID": 2,
                ...
            },
            {
                "Cycle ID": 3,
                ...
            }
            ...
        ]
    }
                
    """
    
    #import data
    D = pd.read_csv(filename, sep="\t|,|;", engine='python')

    #Here I identify the cycles in the data. 
    #Here I grab the rows of the Cap_Chg column that are NOT (nan), extract the indices, and place them in a list
    #Note that the indices follow the index of D, since C is a sub-dataframe
    Cycle_index = D['Cycle ID'][~D['Cycle ID'].isnull()].index.tolist()
    
    #I create a dict that will be transformed into json later.
    data = {}
    data['Cycles'] = []
    
    #Here I loop through the Cycles, and within each cycles I loop throguh the steps
    for [Cyc_i, C_ind] in enumerate(Cycle_index):
        
        #this is the single line data for this cycle
        C_Data = D.iloc[C_ind]
        
        #I put the single line data into a cell of the dictionary strucutre.
        data['Cycles'].append({
            'Cycle ID':                         int(C_Data['Cycle ID']),
            'Cap_Chg(mAh)':                     float(C_Data['Cap_Chg(mAh)']),
            'Cap_DChg(mAh)':                    float(C_Data['Cap_DChg(mAh)']),
            'Specific Capacity-Chg(mAh/g)':     float(C_Data['Specific Capacity-Chg(mAh/g)']),
            'Specific Capacity-Dchg(mAh/g)':    float(C_Data['Specific Capacity-Dchg(mAh/g)']),
            'Chg/DChg Efficiency(%)':           float(C_Data['Chg/DChg Efficiency(%)']),
            'Engy_Chg(mWh)':                    float(C_Data['Engy_Chg(mWh)']),
            'Engy_DChg(mWh)':                   float(C_Data['Engy_DChg(mWh)']),
            'REngy_Chg(mWh/g)':                 float(C_Data['REngy_Chg(mWh/g)']),
            'REngy_Dchg(mWh/g)':                float(C_Data['REngy_Dchg(mWh/g)']),
            'CC_Chg_Ratio(%)':                  float(C_Data['CC_Chg_Ratio(%)']),
            'CC_Chg_Cap(mAh)':                  float(C_Data['CC_Chg_Cap(mAh)']),
            'Plat_Cap(mAh)':                    float(C_Data['Plat_Cap(mAh)']),
            'Plat_Capacity Density(mAh/g)':     float(C_Data['Plat_Capacity Density(mAh/g)']),
            'Plat_Cap_Efficiency(%)':           float(C_Data['Plat_Cap_Efficiency(%)']),
            'Plat_Time(h:min:s.ms)':            str(C_Data['Plat_Time(h:min:s.ms)']),
            'Capacity_Chg(F)':                  float(C_Data['Capacity_Chg(F)']),
            'Capacity_DChg(F)':                 float(C_Data['Capacity_DChg(F)']),
            'IR(O)':                            float(C_Data['IR(O)']),
            'Mid_value Voltage(V)':             float(C_Data['Mid_value Voltage(V)']),
            'Discharge Fading Ratio(%)':        float(C_Data['Discharge Fading Ratio(%)']),
            'Charge Time(h:min:s.ms)':          str(C_Data['Charge Time(h:min:s.ms)']),
            'Discharge Time(h:min:s.ms)':       str(C_Data['Discharge Time(h:min:s.ms)']),
            'Charge IR(O)':                     float(C_Data['Charge IR(O)']),
            'Discharge IR(O)':                  float(C_Data['Discharge IR(O)']),
            'End Temperature(C)':               float(C_Data['End Temperature(�C)']),
            'Net Cap_DChg(mAh)':                float(C_Data['Net Cap_DChg(mAh)']),
            'Net Engy_DChg(mWh)':               float(C_Data['Net Engy_DChg(mWh)']),
            'Energy Efficiency(%)':             float(C_Data['Energy Efficiency(%)']),
            })
        
        #now i construct a sub-dataframe only covering the current cycle, except for the cycle header
        if Cycle_index[Cyc_i]==Cycle_index[-1]:
            C = D.iloc[C_ind+1:]
        else:
            C = D.iloc[C_ind+1:Cycle_index[Cyc_i+1]]
        
        #Here I identify the steps in the cycle. 
        #Because of the multilayered header in the file, the step ID ends up in the Cap_Chg column.
        #Here I grab the rows of the Cap_Chg column that are NOT (nan), extract the indices, and place them in a list
        #Note that the indices follow the index of D, since C is a sub-dataframe
        Step_index = C['Cap_Chg(mAh)'][~C['Cap_Chg(mAh)'].isnull()].index.tolist()
        
        #initializing array of steps
        data['Cycles'][Cyc_i]['Steps']=[]
        
        #Here I loop through the Steps.
        for [Step_i, S_ind] in enumerate(Step_index):
        
            #this is the single line data for this step
            S_Data = D.iloc[S_ind]
            
            #I put the single line data into a cell of the dictionary strucutre. 
            # the wonky column headers are because of the multilayered header in the file
            data['Cycles'][Cyc_i]['Steps'].append({
                'Step ID':                      int(S_Data['Cap_Chg(mAh)']),
                'Step Name':                    str(S_Data['Cap_DChg(mAh)']),
                'Step Time(h:min:s.ms)':        str(S_Data['Specific Capacity-Chg(mAh/g)']),
                'Step Capacity(mAh)':           float(S_Data['Specific Capacity-Dchg(mAh/g)']),
                'Step Capacity Density(mAh/g)': float(S_Data['Chg/DChg Efficiency(%)']),
                'Step Energy(mWh)':             float(S_Data['Engy_Chg(mWh)']),
                'CmpEngergy(mWh/g)':            float(S_Data['Engy_DChg(mWh)']),
                'Capacitance(F)':               float(S_Data['REngy_Chg(mWh/g)']),
                'Start Voltage(V)':             float(S_Data['REngy_Dchg(mWh/g)']),
                'End Voltage(V)':               float(S_Data['CC_Chg_Ratio(%)']),
                'Start Temperature(C)':         float(S_Data['CC_Chg_Cap(mAh)']),
                'End Temperature(C)':           float(S_Data['Plat_Cap(mAh)']),
                'End Capacity(mAh)':            float(S_Data['Plat_Capacity Density(mAh/g)']),
                'Charge Mid-Vol(V)':            float(S_Data['Plat_Cap_Efficiency(%)']),
                'Discharge Mid-Vol(V)':         float(S_Data['Plat_Time(h:min:s.ms)']),
                'DCIR(O)':                      str(S_Data['Capacity_Chg(F)']),
                'ChargeCap(mAh)':               float(S_Data['Capacity_DChg(F)']),
                'DischargeCap(mAh)':            float(S_Data['IR(O)']),
                'Step Engy_Chg(mWh)':           float(S_Data['Mid_value Voltage(V)']),
                'Step Engy_DChg(mWh)':          float(S_Data['Discharge Fading Ratio(%)']),
                'Net DischargeCap(mAh)':        float(S_Data['Charge Time(h:min:s.ms)']),
                'Net Engy_DChg(mWh)':           float(S_Data['Discharge Time(h:min:s.ms)']),
                'Raw Step ID':                  int(S_Data['Charge IR(O)']),
                })
                
            
            #Here i construct a sub-dataframe only covering the current step
            #The Step starts at the row below its header-index
            StartOfStep=S_ind+1
            
            #The step can end at the start of the next step, the start of the next cycle, or at the end of the data, depending on the step.
            
            #if the step is the last in the cycle:
            if Step_index[Step_i]==Step_index[-1]:
                
                #if the cycle is the last cycle
                if Cycle_index[Cyc_i]==Cycle_index[-1]:
                    #The step ends at the end of the file
                    EndOfStep=len(D)
                    
                #if its not the last cycle    
                else:
                    #The step ends at the end of the cycle
                    EndOfStep=Cycle_index[Cyc_i+1]
            #If its not the last step
            else:
                #The step ends at the start of the next step
                EndOfStep=Step_index[Step_i+1]
            
    
            S = D.iloc[StartOfStep:EndOfStep]
                
            
            #Here I add the measurement data in each of the stepts
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Record ID']                 = list(map(int,S['Unnamed: 3']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Time(h:min:s.ms)']          = list(map(str,S['Unnamed: 5']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Voltage(V)']                = list(map(float,S['Unnamed: 7']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Current(mA)']               = list(map(float,S['Unnamed: 9']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Temperature(C)']            = list(map(float,S['Unnamed: 11']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Capacity(mAh)']             = list(map(float,S['Unnamed: 13']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Capacity Density(mAh/g)']   = list(map(float,S['Unnamed: 15']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Energy(mWh)']               = list(map(float,S['Unnamed: 17']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['CmpEng(mWh/g)']             = list(map(float,S['Unnamed: 19']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Realtime']                  = list(map(str,S['Unnamed: 21']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Min-T(C)']                  = list(map(float,S['Unnamed: 23']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Max-T(C)']                  = list(map(float,S['Unnamed: 25']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Avg-T(C)']                  = list(map(float,S['Unnamed: 27']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Power(mW)']                 = list(map(float,S['Unnamed: 29']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Capacitance_Chg(mAh)']      = list(map(float,S['Unnamed: 31']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Capacitance_DChg(mAh)']     = list(map(float,S['Unnamed: 33']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Engy_Chg(mWh)']             = list(map(float,S['Unnamed: 35']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Engy_DChg(mWh)']            = list(map(float,S['Unnamed: 37']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['dQ/dV(mAh/V)']              = list(map(float,S['Unnamed: 39']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['dQm/dV(mAh/V.g)']           = list(map(float,S['Unnamed: 41']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Contact IR(O)']             = list(map(float,S['Unnamed: 43']))
            
    
    #writing to json file if Output_JSON_file=True
    if Output_JSON_file:
        
        #setting wether it should include indentation, to be readable.
        #The size of the file can increase substantially if indents are included.
        if Pretty_JSON:
            json_object = json.dumps(data, indent = 4, ignore_nan=True)
        else:
            json_object = json.dumps(data, ignore_nan=True)
            
        #The name of the file is set here. if no filename is added, it just replacing the .csv with .json
        if outfile=='':
            outfile = filename.replace('.csv', '.json')
        
        #writing the file
        with open(outfile, "w") as outfile:
            outfile.write(json_object)
    
    #the dictionary object is output.
    return data
        
def MITS_2json(filename, outfile='', Output_JSON_file=True, Pretty_JSON=False, CSV=False):
    
    if CSV:
        D = pd.read_csv(filename, sep=",", engine='python', index_col=False)
    else:
        D = pd.read_excel(filename, sheet_name = 1)
    
    #The cycles in the data file
    Cycle_index = D['Cycle_Index'].unique().tolist()

    #I create a dict that will be transformed into json later.
    data = {}
    #initializing array of cycles
    data['Cycles'] = []
    
    #this counter is for keeping track of the Cycle duration
    Cycle_time_counter=0
    
    #Here I loop through the Cycles, and within each cycles I loop throguh the steps
    for [Cyc_i, C_ind] in enumerate(Cycle_index):
            
        #this is the data for this cycle
        C_Data = D.loc[D['Cycle_Index']==C_ind]
    
        #I put the single line data into a cell of the dictionary strucutre.
        data['Cycles'].append({
            'Cycle_ID':                         int(C_ind),
            'Charge_Current(mA)':               float(C_Data['Current(A)'].max()*1e3),
            'Discharge_Current(mA)':            float(C_Data['Current(A)'].min()*1e3),
            'Voltage_High(V)':                  float(C_Data['Voltage(V)'].max()),
            'Voltage_Low(V)':                   float(C_Data['Voltage(V)'].min()),
            'Charge_Capacity(mAh)':             float(C_Data['Charge_Capacity(Ah)'].max()*1e3),
            'Discharge_Capacity(mAh)':          float(C_Data['Discharge_Capacity(Ah)'].max()*1e3),
            'Cycle_Duration(s)':                float(C_Data['Test_Time(s)'].max() - Cycle_time_counter),
            'Start_Time':                       str(C_Data['Date_Time'].min())
            })
        
        #This is the end time of the current cycle. it will be subtracted from the next cycle's end time to get its duration.
        Cycle_time_counter = C_Data['Test_Time(s)'].max()
        
        
        #The steps in the Cycle
        Step_index = C_Data['Step_Index'].unique().tolist()
        
        #initializing array of steps
        data['Cycles'][Cyc_i]['Steps']=[]
        
        #Here I loop through the steps within 
        for [Step_i, S_ind] in enumerate(Step_index):
            
            #this is the single line data for this cycle
            S_Data = C_Data.loc[C_Data['Step_Index']==S_ind]
            
            
            #I put the single line data into a cell of the dictionary strucutre.
            data['Cycles'][Cyc_i]['Steps'].append({
                'Step_ID':                          int(S_ind),
                'Current_High(mA)':                 float(S_Data['Current(A)'].abs().max()*1e3),
                'Current_Low(mA)':                  float(S_Data['Current(A)'].abs().min()*1e3),
                'Voltage_High(V)':                  float(C_Data['Voltage(V)'].max()),
                'Voltage_Low(V)':                   float(C_Data['Voltage(V)'].min()),
                'Charge_Capacity(mAh)':             float(S_Data['Charge_Capacity(Ah)'].max()*1e3-S_Data['Charge_Capacity(Ah)'].min()*1e3),
                'Discharge_Capacity(mAh)':          float(S_Data['Discharge_Capacity(Ah)'].max()*1e3-S_Data['Discharge_Capacity(Ah)'].min()*1e3),
                'Step_Duration(s)':                 str(S_Data['Step_Time(s)'].max()),
                'Start_Time':                       str(S_Data['Date_Time'].min())
                })
            
            
            
            #Here I add the measurement data in each of the stepts
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Data_Point']                 = list(map(int,S_Data['Data_Point']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Cycle_Index']                = list(map(int,S_Data['Cycle_Index']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Step_Index']                 = list(map(int,S_Data['Step_Index']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Date_Time']                  = list(map(str,S_Data['Date_Time']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Test_Time(s)']               = list(map(float,S_Data['Test_Time(s)']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Step_Time(s)']               = list(map(float,S_Data['Step_Time(s)']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Current(A)']                 = list(map(float,S_Data['Current(A)']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Voltage(V)']                 = list(map(float,S_Data['Voltage(V)']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Power(W)']                   = list(map(float,S_Data['Power(W)']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Charge_Capacity(Ah)']        = list(map(float,S_Data['Charge_Capacity(Ah)']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Discharge_Capacity(Ah)']     = list(map(float,S_Data['Discharge_Capacity(Ah)']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Charge_Energy(Wh)']          = list(map(float,S_Data['Charge_Energy(Wh)']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Discharge_Energy(Wh)']       = list(map(float,S_Data['Discharge_Energy(Wh)']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['ACR(Ohm)']                   = list(map(float,S_Data['ACR(Ohm)']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['dV/dt(V/s)']                 = list(map(float,S_Data['dV/dt(V/s)']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['Internal_Resistance(Ohm)']   = list(map(float,S_Data['Internal_Resistance(Ohm)']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['dQ/dV(Ah/V)']                = list(map(float,S_Data['dQ/dV(Ah/V)']))
            data['Cycles'][Cyc_i]['Steps'][Step_i]['dV/dQ(V/Ah)']                = list(map(float,S_Data['dV/dQ(V/Ah)']))

    #writing to json file if Output_JSON_file=True
    if Output_JSON_file:
        
        #setting wether it should include indentation, to be readable.
        #The size of the file can increase substantially if indents are included.
        if Pretty_JSON:
            json_object = json.dumps(data, indent = 4, ignore_nan=True)
        else:
            json_object = json.dumps(data, ignore_nan=True)
            
        #The name of the file is set here. if no filename is added, it just replacing the .csv with .json
        if outfile=='':
            if CSV:
                outfile = filename.replace('.CSV', '.json')
            else:
                outfile = filename.replace('.xlsx', '.json')
        
        
        #writing the file
        with open(outfile, "w") as outfile:
            outfile.write(json_object)
    
    #the dictionary object is output.
    return data        
        
def get_BatchInfo(BatchLabel, DataDirectory = 'Data/', filepath = ""):
    
    """
    BatchInfo = get_BatchInfo(BatchLabel, DataDirectory)
    
    This function returns the info from [DataDirectory]/Data_[BatchLabel]/Batch_[BatchLAbel].json as a python dict,  
    """
    if filepath:
        BatchPath = filepath
    else:
        BatchFolder = 'Data_'+BatchLabel
        BatchDirectory = os.path.join(DataDirectory, BatchFolder)
        
        BatchFile = 'Batch_' + BatchLabel + '.json'
        BatchPath = os.path.join(BatchDirectory, BatchFile)
    
    with open(BatchPath) as file:
        BatchInfo=json.load(file)
    
    
    return BatchInfo
      
def get_SampleInfo(BatchLabel, SampleID, DataDirectory = 'Data/', filepath = ""):
    
    """
    SampleInfo, BatchInfo = get_SampleInfo(BatchLabel, SampleID, DataDirectory)
    
    This function returns the info for a sample [SampleID] from [DataDirectory]/Data_[BatchLabel]/Batch_[BatchLAbel].json as a python dict, 
    along with the info for the full batch, which contains information about all and each sample as well.
    
    """
    
    BatchInfo=get_BatchInfo(BatchLabel, DataDirectory, filepath = filepath)
    
    try:
        SampleInfo = BatchInfo['Samples'][str(SampleID)]
    except KeyError: 
        print('Batch {} contains no Sample {}. The following samples exist: {}'.format(BatchLabel, SampleID, list(BatchInfo['Samples'])))
        SampleInfo = {}
    
    return SampleInfo, BatchInfo

def get_SampleData(BatchLabel, SampleID, DataDirectory = 'Data/', filepath = ""):
    
    """
    SampleData = get_SampleData(BatchLabel, SampleID, DataDirectory):
    
    This script imports the data from [DataDirectory]/Data_[BatchLabel]/CoinCell_[SampleID]_[BatchLabel].json as a python dict
    
    """
    if filepath:
        SamplePath = filepath
    else:
        BatchFolder = 'Data_' + BatchLabel
        BatchDirectory = os.path.join(DataDirectory, BatchFolder)
        
        SampleFile = 'CoinCell_' + str(SampleID) + '_' + BatchLabel + '.json'
        SamplePath = os.path.join(BatchDirectory, SampleFile)
    
    with open(SamplePath) as file:
        SampleData=json.load(file)
    
    return SampleData

def get_CycleData(BatchLabel, SampleID, DataDirectory = 'Data/', Properties=[], filepath = ""):   
    
    """
    Cycle_df = get_CycleData(BatchLabel, SampleID, DataDirectory, Properties=[])
        
    This function extracts the cycle-level data from [DataDirectory]/Data_[BatchLabel]/CoinCell_[SampleID]_[BatchLabel].json
    and returns it as a pandas dataframe.
    
    The Properties argument specifies what properties should be extracted. The default is to extract all properties.
    
    """
    
    SampleData = get_SampleData(BatchLabel, SampleID, DataDirectory, filepath = filepath)    
    
    #Here I'm collecting all the keys under 'Cyclse'
    AllKeys = list(SampleData['Cycles'][0])
    
    #Here I'm selecting the keys that are single valued (ie, not lists).
    CycleDataKeys = []
    for key in AllKeys:
        if not isinstance(SampleData['Cycles'][0][key], list):
            CycleDataKeys.append(key)
    
    
    
    #If no properties were specified, all of them are assumed
    #Here I also make sure that the selected keys are valid.
    if Properties == []:
        Properties = CycleDataKeys
    else:
        nonkeys=[]
        for key in Properties:
            if key not in CycleDataKeys:
                nonkeys.append(key)
        if nonkeys:
            raise KeyError('{0} Do not exist. Try these: {1}'.format(nonkeys, CycleDataKeys))
    
    #Here I go through the file and collect all the data into a list of lists, and then packaging them as a dataframe.
    #starting a list of lists to make a pandas data frame later.
    data=[]
    NCyc = len(SampleData['Cycles'])
    for Cyc_i in range(NCyc):
        Cyc_vals=[]
        for key in Properties:
            key_val = SampleData['Cycles'][Cyc_i][key]
            Cyc_vals.append(key_val)
        
        data.append(Cyc_vals)
    
    Cycle_df = pd.DataFrame(data, columns = Properties)

    return Cycle_df

def get_StepData(BatchLabel, SampleID, DataDirectory = 'Data/', Properties=[], filepath = ""):   
        
    """
    Step_df = get_StepData(BatchLabel, SampleID, DataDirectory, Properties=[])
        
    This function extracts the step-level data from [DataDirectory]/Data_[BatchLabel]/CoinCell_[SampleID]_[BatchLabel].json
    and returns it as a pandas dataframe.
    
    The Properties argument specifies what properties should be extracted. The default is to extract all properties.
    
    """
    
    SampleData = get_SampleData(BatchLabel, SampleID, DataDirectory, filepath = filepath) 
    
    #Here I'm collecting all the keys under 'Steps'
    AllKeys = list(SampleData['Cycles'][0]['Steps'][0])
    
    #Here I'm selecting the keys that are single valued (ie, not lists).
    StepDataKeys = []
    for key in AllKeys:
        if not isinstance(SampleData['Cycles'][0]['Steps'][0][key], list):
            StepDataKeys.append(key)
    
    
    #If no properties were specified, all of them are assumed
    #Here I also make sure that the selected keys are valid.
    if Properties == []:
        Properties = StepDataKeys
    else:
        nonkeys=[]
        for key in Properties:
            if key not in StepDataKeys:
                nonkeys.append(key)
        if nonkeys:
            raise KeyError('{0} Do not exist. Try these: {1}'.format(nonkeys, StepDataKeys))
    
    #Here I go through the file and collect all the data into a list of lists, and then packaging them as a dataframe.
    #starting a list of lists to make a pandas data frame later.
    data=[]
    NCyc = len(SampleData['Cycles'])
    for Cyc_i in range(NCyc):
        NStep = len(SampleData['Cycles'][Cyc_i]['Steps'])
        for Step_i in range(NStep):
            Step_vals=[]
            for key in Properties:
                key_val = SampleData['Cycles'][Cyc_i]['Steps'][Step_i][key]
                Step_vals.append(key_val)
                
            data.append(Step_vals)
    
    
    Step_df = pd.DataFrame(data, columns = Properties)
    
    return Step_df

def get_PointData(BatchLabel, SampleID, DataDirectory = 'Data/', Properties=[], filepath = ""):   
    
    """
    Point_df = get_PointData(BatchLabel, SampleID, DataDirectory, Properties=[])
        
    This function extracts the point-level data from [DataDirectory]/Data_[BatchLabel]/CoinCell_[SampleID]_[BatchLabel].json
    and returns it as a pandas dataframe.
    
    The Properties argument specifies what properties should be extracted. The default is to extract all properties.
    
    """
    
    SampleData = get_SampleData(BatchLabel, SampleID, DataDirectory, filepath = filepath)  
    
    #Here I'm collecting all the keys under 'Steps'
    AllKeys = list(SampleData['Cycles'][0]['Steps'][0])
    
    #Here I'm selecting the keys that are lists.
    PointDataKeys = []
    for key in AllKeys:
        if isinstance(SampleData['Cycles'][0]['Steps'][0][key], list):
            PointDataKeys.append(key)
    
    if Properties == []:
        Properties = PointDataKeys
    else:
        nonkeys=[]
        for key in Properties:
            if key not in PointDataKeys:
                nonkeys.append(key)
        if nonkeys:
            raise KeyError('{0} Do not exist. Try these: {1}'.format(nonkeys, PointDataKeys))
    
    
    
    #starting an empty pandas dataframe to which I then append the data stepwise.
    Point_df = pd.DataFrame(data=[], columns = Properties)
    NCyc = len(SampleData['Cycles'])
    for Cyc_i in range(NCyc):
        NStep = len(SampleData['Cycles'][Cyc_i]['Steps'])
        for Step_i in range(NStep):
            temp_df = pd.DataFrame()
            for key in Properties:
                temp_df[key] = SampleData['Cycles'][Cyc_i]['Steps'][Step_i][key]
                 
            Point_df = pd.concat([Point_df, temp_df], ignore_index=True)
    
    return Point_df
            
def get_SampleList(DataDirectory = 'Data/'):
    
    """
    SampleList = get_SampleList(DataDirectory)
    This function returns a dict listing all samples.
    """
    
    SampleList={}
    
    regexp=r"^Data_(.*)$"
    
    
    for root, dirs, files in os.walk(DataDirectory):
        folder = os.path.split(root)[1]
        
        FolderMatch = re.findall(regexp, folder)
        #print(FolderMatch)
        if FolderMatch:
            BatchLabel = FolderMatch[0]
            BI = get_BatchInfo(BatchLabel, DataDirectory)
            Samples = list(BI['Samples'])
            SampleList[BatchLabel]=Samples

            
    return SampleList
    
def get_BatchCycles(SampleList, DataDirectory = 'Data/', CapacityNorm_CycleID = 2):
    
    
    dfBatch = pd.DataFrame()

    for BatchLabel in list(SampleList.keys()):
        for SampleID in SampleList[BatchLabel]:
        
            SampleInfo, BatchInfo = get_SampleInfo(BatchLabel, SampleID, DataDirectory)
        
            Thickness = SampleInfo['ECCthickness'] - BatchInfo['CurrentCollector']['Thickness'] #micrometer
            Mass = (SampleInfo['ECCmass']-BatchInfo['CurrentCollector']['Mass'])*1e-3 #grams
            Slurry_Mass = BatchInfo['Slurry']['AM']['Mass'] + BatchInfo['Slurry']['Binder']['Mass']*BatchInfo['Slurry']['Binder']['Binder_Concentration'] + BatchInfo['Slurry']['Carbon']['Mass'] #grams
            AM_Mass_frac = BatchInfo['Slurry']['AM']['Mass']/Slurry_Mass
            AM_Mass = Mass * AM_Mass_frac #grams
            
            CCd = BatchInfo['CurrentCollector']['Diameter[cm]'] #cm
            area = (CCd/2)**2*pi #cm2
        
            df = get_CycleData(BatchLabel, SampleID, DataDirectory, Properties=['Cycle_ID', 'Discharge_Current(mA)', 'Discharge_Capacity(mAh)'])
            
            df.loc[:,'Current(mA/cm2)'] = -df.loc[:,'Discharge_Current(mA)']/area
            df.loc[:,'Capacity(mAh/cm2)'] = df.loc[:,'Discharge_Capacity(mAh)']/area
            df.loc[:,'Capacity(mAh/gAM)'] = df.loc[:,'Discharge_Capacity(mAh)']/AM_Mass
            df.loc[:,'C-rate(1/h)'] = df.loc[:,'Current(mA/cm2)']/df.loc[df['Cycle_ID']==CapacityNorm_CycleID,'Capacity(mAh/cm2)'].values[0]
        
            #I'm dropping the initial columns.
            df.drop(columns = ['Discharge_Current(mA)', 'Discharge_Capacity(mAh)'], inplace = True)
            
        
            df.loc[:,'Thickness(um)'] = Thickness
            df.loc[:,'Cycler_Program'] = SampleInfo['Cycler_Program']
            df.loc[:,'Wet_Thickness(um)'] = SampleInfo['DoctorBlade']
            df.loc[:,'Cathode'] = BatchInfo['Slurry']['AM']['Material']
            df.loc[:,'SampleID'] = SampleID
            df.loc[:,'Batch'] = BatchLabel
        
            dfBatch = dfBatch.append(df, ignore_index=True)

    
    return dfBatch

def init_OtherData():
    
    filename='/home/martin/Work/Research/Batteries/Thickness_Effect/Data/Other/DYW_ZH_Data.csv'
    D = pd.read_csv(filename)
    
    
    dfnew = pd.DataFrame()
    
    dfnew.loc[:,'Batch']                    = D['Source'] + ', ' + D['Cathode']
    dfnew.loc[:,'C-rate(1/h)']              = D['C']
    dfnew.loc[:,'Thickness(um)']            = D['t[mu]']
    dfnew.loc[:,'Cycler_Program']           = D['Source'] + ', ' + D['Cathode']
    dfnew.loc[:,'Cathode']                  = D['Cathode']
    dfnew.loc[:,'SampleID']                 = D['t[mu]']
    dfnew.loc[:,'Cycle_ID']                 = np.nan
    dfnew.loc[:,'Wet_Thickness(um)']        = np.nan
    
    index = D['Source'] == 'H. Zheng (2012)'
    dfnew.loc[index,'Capacity(mAh/cm2)']        = D.loc[index,'Qa[mAh/cm2]']
    
    index = D['Source'] == 'D.Y.W. Yu (2006)'
    rho=2.2
    dfnew.loc[index,'Capacity(mAh/gAM)']        = D.loc[index,'Qm[mAh/gAM]']
    dfnew.loc[index,'Capacity(mAh/cm2)']        = D.loc[index,'Qm[mAh/gAM]']*rho*(D.loc[index,'t[mu]']*1e-4)

    
    
    #below I calculate 'Capacity(mAh/gAM)' and 'Current(mA/cm2)' for the two data sets
    
    #AM loadings to calculate mass capacity of H. Zheng
    AMl_ZH = np.array([5.56, 11.48, 17.52, 24.01, 1.51, 3.64, 7.31, 11.34, 15.83]) #mg/cm2
    tt_ZH = np.array([24, 50, 76, 104, 10, 25, 50, 77, 108]) #um
    cat_ZH = ['NMC', 'NMC', 'NMC', 'NMC', 'LFP', 'LFP', 'LFP', 'LFP', 'LFP'] 
    
    for i, ti in enumerate(tt_ZH):
        index = (dfnew['Batch'] == 'H. Zheng (2012), ' + cat_ZH[i]) & (dfnew['Thickness(um)'] == ti) & (dfnew['Cathode'] == cat_ZH[i])
        
        dfnew.loc[index,'Capacity(mAh/gAM)'] = dfnew.loc[index,'Capacity(mAh/cm2)']/(AMl_ZH[i])
        
        cap_01C = dfnew.loc[(index) & (dfnew['C-rate(1/h)']==0.1),'Capacity(mAh/cm2)'].mean()
        dfnew.loc[index,'Current(mA/cm2)'] = dfnew.loc[index,'C-rate(1/h)']*cap_01C
    
    
    index = (dfnew['Batch'] == 'D.Y.W. Yu (2006), LFP') 
    tt_DYW = dfnew.loc[index, 'Thickness(um)'].unique()
    for ti in tt_DYW:
        index = (dfnew['Batch'] == 'D.Y.W. Yu (2006), LFP') & (dfnew['Thickness(um)'] == ti)
        cap_01C = dfnew.loc[(index) & (dfnew['C-rate(1/h)']==0.1),'Capacity(mAh/cm2)'].mean()
        dfnew.loc[index,'Current(mA/cm2)'] = dfnew.loc[index,'C-rate(1/h)']*cap_01C 
        
    return dfnew

def CRate_groups(df, ProgDict):

    for i, Cyc_Prog in enumerate(list(ProgDict.keys())):
        
        crates = ProgDict[Cyc_Prog]
        
        
        bins = crates[:-1] + np.diff(crates)/2
        bins = np.append(bins, crates[-1] + np.diff(crates)[-1]/2)
        bins = np.insert(bins, 0, 0)
        
        for i, cr in enumerate(crates):
            
            Crate_group = (df.loc[:,'C-rate(1/h)']>bins[i]) & (df.loc[:,'C-rate(1/h)']<=bins[i+1]) & (df.loc[:,'Cycler_Program'] == Cyc_Prog)
            df.loc[Crate_group,'C-rate(rnd)'] = cr
            df.loc[Crate_group,'C-rate(rnd-frac)'] = df.loc[Crate_group,'C-rate(1/h)']/cr-1
            df.loc[Crate_group,'C-rate(mean)'] = df.loc[Crate_group,'C-rate(1/h)'].mean()
            df.loc[Crate_group,'C-rate(std)'] = df.loc[Crate_group,'C-rate(1/h)'].std()
            
            thicks = np.sort(df.loc[df['C-rate(rnd)'] == cr,'Thickness(um)'].unique())
            
            for tt in thicks:
                index = (df['C-rate(rnd)'] == cr) & (df['Thickness(um)'] == tt)
                df.loc[index,'Avg_DCapacity(mAh/cm2)'] = df.loc[index,'Capacity(mAh/cm2)'].mean()
                df.loc[index,'Std_DCapacity(mAh/cm2)'] = df.loc[index,'Capacity(mAh/cm2)'].std()
                df.loc[index,'Avg_DCapacity(mAh/gAM)'] = df.loc[index,'Capacity(mAh/gAM)'].mean()
                df.loc[index,'Std_DCapacity(mAh/gAM)'] = df.loc[index,'Capacity(mAh/gAM)'].std()
                
                df.loc[index,'Avg_Charge_peak(V)'] = df.loc[index,'Charge_peak(V)'].mean()
                df.loc[index,'Std_Charge_peak(V)'] = df.loc[index,'Charge_peak(V)'].std()
                df.loc[index,'Avg_Discharge_peak(V)'] = df.loc[index,'Discharge_peak(V)'].mean()
                df.loc[index,'Std_Discharge_peak(V)'] = df.loc[index,'Discharge_peak(V)'].std()
                df.loc[index,'Avg_Overpotential(V)'] = df.loc[index,'Overpotential(V)'].mean()
                df.loc[index,'Std_Overpotential(V)'] = df.loc[index,'Overpotential(V)'].std()
    return df

def get_CritRL(dfall):
    
    dfCrit = pd.DataFrame()
    
    for name, group in dfall.groupby(by=['Batch']):    
        crts = np.sort(group['C-rate(rnd)'].dropna().unique())
    
        df = pd.DataFrame()
        df['C-rate(rnd)'] = crts
        df['Batch'] = name
        df['Cathode'] = group['Cathode'].values[0]
    
    
        for cr in crts:
        
            dfcp = group[group['C-rate(rnd)'] == cr].copy()
            
            df.loc[df['C-rate(rnd)'] == cr,'C-rate_mean(1/h)'] = dfcp['C-rate(1/h)'].mean()
            df.loc[df['C-rate(rnd)'] == cr,'C-rate_std(1/h)'] = dfcp['C-rate(1/h)'].std()
        
            index = dfcp['Avg_DCapacity(mAh/cm2)'] == dfcp['Avg_DCapacity(mAh/cm2)'].max()
            maxCap = dfcp.loc[index,'Avg_DCapacity(mAh/cm2)'].values[0]
            
            index = dfcp['Avg_DCapacity(mAh/cm2)']==maxCap
            maxCapTh = dfcp.loc[index, 'Thickness(um)'].values[0]
           
            df.loc[df['C-rate(rnd)'] == cr,'Avg_DCapacity_max(mAh/cm2)'] = maxCap
            df.loc[df['C-rate(rnd)'] == cr,'Thickness_max(um)'] = maxCapTh
            
            dfcptt = dfcp['Thickness(um)'].unique()
            #calculating error bar of the critical thickness. taking halfway to adjacent points
            temp = (dfcptt[dfcptt != maxCapTh] - maxCapTh)/2
            
            if temp[temp<0].size>0:
                crit_lo = -temp[temp<0].max()
            else:
                crit_lo = np.nan
                
            if temp[temp>0].size>0:
                crit_hi = temp[temp>0].min()
            else:
                crit_hi = np.nan
            
            df.loc[df['C-rate(rnd)'] == cr,'Thickness_lo(um)'] = crit_lo
            df.loc[df['C-rate(rnd)'] == cr,'Thickness_hi(um)'] = crit_hi
            
        dfCrit = dfCrit.append(df, ignore_index=True)
        
    return dfCrit

def get_overpotential(Batch, Sample, Plot=0, Curve_Volt = 0.05, Incl_Cyc = np.array([]), Dcap_Lim = 0.3, filter_V_lim = (4.1, 4.38), dqdv2_filter_lim = 4.25):
    
    
    if Plot:
        fig, ax1= plt.subplots(figsize=(13,6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        if Plot == 2: 
            ax2 = ax1.twinx()
            
        
    df = get_PointData(Batch, Sample)
    df_cyc = get_CycleData(Batch, Sample)
    
    OPdf = pd.DataFrame(columns=['Cycle_ID', 'Charge_peak(V)', 'Discharge_peak(V)', 'Overpotential(V)'])
    OPdf.loc[:, 'Cycle_ID'] = df_cyc['Cycle_ID'].unique()
    if ~Incl_Cyc.any():
        Incl_Cyc = df_cyc['Cycle_ID'].unique()
    
    groups = df.groupby(by=['Cycle_Index','Step_Index'])
    
    dcap=np.array([])
    
    for name, group in groups:
        
        TotCap = group.loc[:, 'Current(A)'].sum() 
        if TotCap > 0:
            mode = "Charge"
        elif TotCap < 0:
            mode = "Discharge"
        else:
            mode = "Rest"    
        
        #I only unclude cycles that have some minimumum capacity
        #that minimum is a fraction(Dcap_Lim=0.3) of the initial cycle capacity
        Dcap = df_cyc.loc[df_cyc['Cycle_ID']==name[0],'Discharge_Capacity(mAh)'].values
        dcap = np.append(dcap,Dcap)
        
        #var = pd.DataFrame()
        
        if (mode != "Rest") & (Dcap/dcap[0]>Dcap_Lim) & (name[0] in Incl_Cyc):
    
            V = group.loc[:,'Voltage(V)'].to_numpy()
            if mode == 'Charge': 
                Q = group.loc[:,'Charge_Capacity(Ah)'].to_numpy()
            elif mode == 'Discharge': 
                Q = group.loc[:,'Discharge_Capacity(Ah)'].to_numpy()
            
            #V has adjacent points with identical value, this causes np.gradient to throw an ZeroDivisionError. Therefore I'm replacing the corresponding Q values by their average and remove one point.
            
            dV0_index = np.diff(V)==0
            # dV0_index is one element shorter than V and Q. so I'm adding a False value at the end. Then the True values will correspond to the FIRST of the pair V[i]-V[i+1]=0 
            dV0_index1 = np.append(dV0_index,False) #first
            dV0_index2 = np.append(False, dV0_index) #second
            
            #calculating the average Q and placing it in the first of the pairs
            Q[dV0_index1] = (Q[dV0_index1] + Q[dV0_index2])/2
            #here im removing the second element of the pair, both from Q and V.
            Q = np.delete(Q, dV0_index2)
            V = np.delete(V, dV0_index2)
            
            # ! This keeps throwing an ZeroDivisionError even when there is no dV=0
            with np.errstate(divide='ignore'):
                dQdV = np.gradient(Q,V)
            
            #print("dqdv nan: " + str(sum(np.isnan(dQdV))))
            #print("dqdv inf: " + str(sum(abs(dQdV)==np.inf)))
            #removing the "pretty much equal to zero" and 
            index = (abs(dQdV) > 1e-4) & (~np.isnan(dQdV))
            dQdV = dQdV[index]
            V = V[index]
                
            if (mode=='Charge'):
                
                #im setting the SG_window to match the number of points within a Curve_Volt voltage window
                
                #for the charging step, im only evaluating the points around where the peak is
                filter_V = (V>filter_V_lim[0]) & (V<filter_V_lim[1])
                
                
                vtemp = V[filter_V]
                
                if vtemp.any():
                    pts_per_volt = len(vtemp)/(max(vtemp) - min(vtemp))
                    pts_per_xV = pts_per_volt * Curve_Volt
                    #rounding UP to nearest odd
                    SG_window =  int(np.ceil(pts_per_xV) // 2 * 2 + 1)                        
                else:
                    SG_window = np.nan
                
                #print(SG_window)
                
                
                if sum(index) >= SG_window:
                    
                    #print("Charge : " + str(SG_window))
                    
                    dQdV_smooth = savgol_filter(dQdV,SG_window,2)
                    
                    #After filtering, the edge points are crazy and they can't be excluded with just a voltage limit. so after the voltage limit, I'm excluding all points after the first one with d(dqdv)/d(v) larger than one standard deviation.
                    vtemp_V = V[filter_V]
                    dqdvtemp_V = dQdV_smooth[filter_V]
                    dqdv2_V = np.gradient(dqdvtemp_V, vtemp_V)
                    
                    
                    #argmax(..) gives the index of the first time dqdv2 exceeeds one std away from the mean.
                    #In addition, to avoid that this removes the noisier peaks. I'm only considering this filter above a voltage thresh.
                    voltage_thresh_filter = vtemp_V > dqdv2_filter_lim
                    i_dqdv2_filter = np.argmax(abs(dqdv2_V[voltage_thresh_filter]) > dqdv2_V.mean()+dqdv2_V.std())
                    i_dqdv2_filter += np.argmax(voltage_thresh_filter) - 5  #Im removing 5 more points for good measure.
                    dqdv2_filter = np.array([True] * len(vtemp_V))
                    dqdv2_filter[i_dqdv2_filter:] = False
                    
                    #print(i_dqdv2_filter)
                    
                    
                    vtemp = vtemp_V[dqdv2_filter]
                    dqdvtemp = dqdvtemp_V[dqdv2_filter]
                    
                    peak_index = dqdvtemp == dqdvtemp.max()
                    peakV = vtemp[peak_index]
                    OPdf.loc[OPdf['Cycle_ID']==name[0], 'Charge_peak(V)'] = peakV
                    
                    if Plot:
                        ax1.plot(vtemp,dqdvtemp, marker='.', linestyle ='-', color = colors[np.mod(name[0],len(colors))])
                        ax1.plot(peakV, dqdvtemp.max(), marker = 'o', markersize = 10, alpha = 0.5, color = 'k')
                        if Plot==2:
                            ax1.plot(vtemp_V,dqdvtemp_V, marker='.', linestyle ='-', color = 'k', zorder = 0)
                            ax2.plot(vtemp_V,dqdv2_V, marker='.', linestyle ='-', color = 'r', zorder = 0)
                            ax2.hlines([dqdv2_V.mean() - dqdv2_V.std(), dqdv2_V.mean(), dqdv2_V.mean() + dqdv2_V.std()], 4.1, 4.4, linestyle = '--', color = 'r')
                            ax2.vlines(dqdv2_filter_lim, min(dqdv2_V), max(dqdv2_V), linestyle = '--', color = 'r')
                        
                    
                    #var['dqdv'] = dqdvtemp
                    #var.loc[:,'v'] = vtemp
                        
            elif (mode=='Discharge'):
                
                    
                vtemp = V
                
                
                if vtemp.any():
                    pts_per_volt = len(vtemp)/(max(vtemp) - min(vtemp))
                    pts_per_xV = pts_per_volt * Curve_Volt
                    #rounding UP to nearest odd
                    SG_window =  int(np.ceil(pts_per_xV) // 2 * 2 + 1)
                    
                    #I only include about 0.4V worth of points
                    #from where the discharge step starts (after rest)
                    
                    index = np.array([False] * len(V))
                    index[:int(pts_per_volt*0.4)] = True
                    
                        
                else:
                    SG_window = np.nan
                
                
                
                
                
                if sum(index) >= SG_window:
                    
                    #print("Discharge : " + str(SG_window))
                    
                    
                    dQdV_smooth = savgol_filter(dQdV,SG_window,2)

                    vtemp = V[index]
                    
                    dqdvtemp = dQdV_smooth[index]
                    peak_index = dqdvtemp == dqdvtemp.min()
                    peakV = vtemp[peak_index]
    
                    OPdf.loc[OPdf['Cycle_ID']==name[0], 'Discharge_peak(V)'] = peakV
                    

                    if Plot:
                        ax1.plot(vtemp,dqdvtemp, marker='.', linestyle ='-', color = colors[np.mod(name[0],len(colors))])
                        ax1.plot(peakV, dqdvtemp.min(), marker = 'o', alpha = 0.5, color = 'k', markersize=10)
    
                
                
    OPdf.loc[:,'Overpotential(V)'] = (OPdf.loc[:,'Charge_peak(V)'] - OPdf.loc[:,'Discharge_peak(V)'])/2
    
    
    
    if Plot:
        ax1.set_ylabel('dQ/dV [Ah/V]', fontsize = 16)
        ax1.set_xlabel('Voltage [V]', fontsize = 16)
        ax1.set_title('B: ' + Batch + ', S: ' + Sample)
        if Plot ==2:
            y0 = 0
            dy1 = 0.04
            ax1.set_ylim((y0-dy1, y0+dy1))
            dy2 = 10
            ax2.set_ylim((y0-dy2, y0+dy2))
        
        plt.show()
    
    return OPdf#, var


def OPSampleList(SampleList, **kwargs):
    
    OPall = pd.DataFrame()
    
    for batch in SampleList.keys():
        for sample in SampleList[batch]:
            OPdf = get_overpotential(batch, sample, **kwargs)
            OPdf['SampleID'] = sample
            OPdf['Batch'] = batch
            
            ('B: ' + batch + ', S: ' + sample)
            OPall = OPall.append(OPdf, ignore_index = True)
            
    
    return OPall
            
    
    
    
    return