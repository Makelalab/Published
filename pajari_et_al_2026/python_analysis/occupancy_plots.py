# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 09:42:31 2026

@author: pajaria3
"""

import pandas as pd
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os


"""
Script that reads from csv-files the diffusional state occupancy values
(in fractions) generated from the Gaussian Mixture Model (GMM). 
Also plotted: a comparison of occupancy values from DnaK-HaloTag GMM.

Requires csv-files with columns for fractional occupancy values/timepoints/state/strain/SEM (standard error of the mean)
"""
 

root = tk.Tk()
root.withdraw()

#Set working directory, files will be saved here
folder_path = filedialog.askdirectory()
os.chdir(folder_path)

#Open and read files
file_path = filedialog.askopenfilenames(title="Select CSV files", filetypes=[("CSV files", "*.csv")])

#Read all experiments into a list of dataframes
dataframes = []
for file in file_path:
    dataframes.append(pd.read_csv(file))
    

# Define a list of three distinct markers (one per state)
markers = ['o', 's', '^']  # Circle, square, and triangle markers
colors = ["blue", "darkorange", "green"]


#Plot

for df in dataframes:
        # Group the data by 'State' and plot each group
        if df["Strain"][0] == "DnaK":
            for i, (state, group) in enumerate(df.groupby('State')):
                marker = markers[i]  # Assign one of the three markers to each state
                color = colors[i]
                plt.errorbar(group['Timepoint'], group['Occupancy'], alpha = 0.5, marker=marker, color = color, linestyle='dashed', label=f'{state}')
            
        else:   
            for i, (state, group) in enumerate(df.groupby('State')):
                marker = markers[i]  # Assign one of the three markers to each state
                plt.errorbar(group['Timepoint'], group['Occupancy'], yerr=group['SEM'], marker=marker, linestyle='-', label=f'{state}')
        
        
        # Customize the plot
        plt.xlabel('Time point')
        plt.ylabel('State Occupancy')
        plt.ylim(0, 1)
        plt.title('Occupancy vs Timepoint by State')
        plt.legend()
        

plt.show()

