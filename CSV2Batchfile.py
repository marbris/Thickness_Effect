
#%%
import pandas as pd


file = 'Data/Data_220428_NMC/inputs.csv'
D = pd.read_csv(file)

for i in range(len(D)):
    SampleID = D.loc[i,'SampleID']
    DoctorBlade = D.loc[i,'DoctorBlade[um]']
    mass = D.loc[i,'mass[mg]']
    thickness = D.loc[i,'thickness[um]']
    
    L='{'
    R='}'
    print(f'\"{SampleID}\": {L}"SampleID": {SampleID}, "DoctorBlade": {DoctorBlade}, "ECCmass": {mass}, "ECCthickness": {thickness}, "Cycler_Channel": null, "Cycler_Program": "EIS_Tortuosity", "StartTime": null{R},')
          
      #    }: { \"SampleID\": , \"DoctorBlade\":1400, \"ECCmass\": 0,' )
          
          
       #   : \{\"SampleID\": {q}, \"DoctorBlade":1400, "ECCmass": 0, "ECCthickness": 0, "Cycler_Channel": "1", "Cycler_Program": "01C_1C", "StartTime": "2022-03-04 16:50"}')
    
    
#%%
q = 2



#%%
