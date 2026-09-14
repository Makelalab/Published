# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 13:19:40 2026

@author: pajaria3
"""

import pandas as pd
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os

"""
Script that reads from csv-files the Hidden Markov model(HMM)-dervied dwell times (in seconds)

Also plotted: a comparison of dwell time values from DnaK-HaloTag HMM.

Requires csv-files with columns for HMM dwell time values/timepoints/state/strain/SEM (standard error of the mean)

Second script plots the influx rates of the DnaK slow state. Same file requirements as above.
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
    

# Define a list of three distinct markers
markers = ['o', 's', '^']  # Circle, square, and triangle markers
colors = ["blue", "darkorange", "green"]

fig, ax = plt.subplots(figsize=(6,4))  # one figure/axes for all plots

for df in dataframes:
        # Group the data by 'State' and plot each group
        if df["Strain"][0] == "DnaK":
            for i, (state, group) in enumerate(df.groupby('State')):
                marker = markers[i]  # Assign one of the three markers to each state
                color = colors[i]
                ax.plot(group['Timepoint'], group['Dwell time'], alpha = 0.5, marker=marker, color = color, linestyle='dashed', label=f'State {state}')
            
        else:   
            for i, (state, group) in enumerate(df.groupby('State')):
                marker = markers[i]  # Assign one of the three markers to each state
                color = colors[i]
                ax.errorbar(group['Timepoint'], group['Dwell time'], yerr=group['SEM'], alpha = 0.5, marker=marker, color = color, linestyle='-', label=f'State {state}')        
        
        # Customize the plot
        plt.ylim(-0.1, 2.0)
        plt.xlabel('Time point')
        plt.ylabel('Dwell time (s)')
        plt.title('Dwell time by state')
        plt.legend()
        

#Show the plot
plt.show()





#Plot DnaK-HaloTag low state influx rates


root = tk.Tk()
root.withdraw()

#Set working directory, files will be saved here
folder_path = filedialog.askdirectory()
os.chdir(folder_path)

file_path = filedialog.askopenfilenames(title="Select CSV files", filetypes=[("CSV files", "*.csv")])

#Read all experiments into a list of dataframes

df = pd.read_csv(file)
    

# Define a list of three distinct markers
markers = ['o', 's', '^']  # Circle, square, and triangle markers
colors = ["blue", "darkorange", "green"]


 
  
for i, (state, group) in enumerate(df.groupby('State')):
    marker = markers[i]  # Assign one of the three markers to each state
    color = colors[i]
    plt.errorbar(group['Timepoint'], group['Influx'], yerr=group['SEM'], alpha = 0.5, marker=marker, color = color, linestyle='-', label=f'State {state}')        

# Customize the plot
plt.ylim(0,12)
plt.xlabel('Time point')
plt.ylabel('Influx')
plt.title('State influx')
plt.legend()
       
#Show the plot
plt.show()



