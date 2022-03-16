#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 15:12:10 2022

@author: martin
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

#%%

a = pd.DataFrame({'Program': ['A', 'A', 'B', 'B', 'Total', 'Total'],
                  'Scenario': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
                  'Duration': [4, 3, 5, 4, 9, 7]})
b = pd.DataFrame({'Program': ['A', 'A', 'B', 'B', 'C', 'C', 'Total', 'Total'],
                  'Scenario': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y'],
                  'Duration': [4, 3, 5, 4, 3, 2, 12, 9]})

unique = a["Program"].append(b["Program"]).unique()
palette = dict(zip(unique, sns.color_palette(n_colors=len(unique))))
palette.update({"Total":"k"})


ax = sns.boxplot(x=b['Scenario'], y=b['Duration'], hue=b['Program'])



#%%

sns.set_theme(style="whitegrid")

tips = sns.load_dataset("tips")

fig, ax= plt.subplots(figsize=(5,4))    


ax = sns.boxplot(x="day", y="total_bill", hue="smoker",data=tips, palette="Set3")


#%%

for i, child in enumerate(ax.properties()['children']):
    if isinstance(child, matplotlib.patches.PathPatch):
        print(i, child)

#%%
mybox = ax.properties()['children'][51]
mybox.set_facecolor('blue')
    
