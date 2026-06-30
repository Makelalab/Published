# -*- coding: utf-8 -*-
"""
Created on Sat May 30 11:30:43 2026

@author: pajaria3
"""
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os
import re

"""
This script calculates the state transition rates/dwell times/influx rates and the
standard error of the mean (SEM) for each parameter from output parameter csv-files resulting from the ExTrack Hidden Markov Model analysis
Requires the parameters output files from ExTrack. Script requires all files per time point (replicates) to calculate SEM.
"""

root = tk.Tk()
root.withdraw()

#Set working directory, files will be saved here
folder_path = filedialog.askdirectory()
os.chdir(folder_path)

#open file
file_path = filedialog.askopenfilenames(title="Select CSV files", filetypes=[("CSV files", "*.csv")])

#Read all experiments into a list of dataframes
dataframes = []
for idx, file in enumerate(file_path):
    data = pd.read_csv(file)
    data = data.T
    data.columns = data.iloc[0]       # use first row as header
    data = data.iloc[1:].reset_index(drop=True)
    data["experiment_id"] = idx
    dataframes.append(data)
    

#frame time
dt = 0.00545


#Calculate transitions rates
for data in dataframes:
    data["p01_rate"] = data["p01"]/ dt
    data["p10_rate"] = data["p10"]/ dt
    data["p20_rate"] = data["p20"]/ dt
    data["p02_rate"] = data["p02"]/ dt
    data["p12_rate"] = data["p12"]/ dt
    data["p21_rate"] = data["p21"]/ dt
    

    #Calculate state influxes 

    data["S0_influx_rate"] = data["p10_rate"] + data["p20_rate"]
    data["S1_influx_rate"] = data["p01_rate"] + data["p21_rate"]
    data["S2_influx_rate"] = data["p02_rate"] + data["p12_rate"]

#Calculate Dwell times for S0/S1/S2 and add to dataframe
for data in dataframes:
    data["S0_dwelltime"] = 1/ (data["p01_rate"] + data["p02_rate"])
    data["S1_dwelltime"] = 1/ (data["p10_rate"] + data["p12_rate"])
    data["S2_dwelltime"] = 1/ (data["p21_rate"] + data["p20_rate"])





# all data from in one dataframe
all_data = pd.concat(dataframes, ignore_index=True)

# Columns to summarize
dwell_cols   = ['S0_dwelltime', 'S1_dwelltime', 'S2_dwelltime']
influx_rates = ['S0_influx_rate', 'S1_influx_rate', 'S2_influx_rate']     
p_cols       = [c for c in all_data.columns if re.fullmatch(r'p\d{2}', c)]
rate_cols    = [c for c in all_data.columns if re.fullmatch(r'p\d{2}_rate', c)]

 #These columns will be included
cols = dwell_cols + influx_rates + p_cols + rate_cols

# Mean and SEM across experiments for each parameter
stats = all_data[cols].agg(['mean', lambda s: s.sem(ddof=1)]).T
stats.columns = ['mean', 'sem']
stats = stats.reset_index().rename(columns={'index': 'parameter'})

# export `stats` as you already do
print(stats.head())


stats.to_csv(f"{folder_path}/your_filename.csv")


