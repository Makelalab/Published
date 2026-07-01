# -*- coding: utf-8 -*-
"""
Created on Thu May  7 22:04:45 2026

@author: pajaria3
"""


import pandas as pd
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
import seaborn as sns
import numpy as np
from scipy import stats
from utils import preprocess_offest
from utils import sem_func




"""
This script calculates the SEM for the tracking files of DnaK 37C replicates and plots the last 60s of the timecourse at 
37C as a scatterplot. Requires tracking csv-files.

"""

#Preprocessing raw individual csv-files if required (each tracking replicate should start from frame 0)
filename = "filename_here"
preprocess_offest(filename)


#frame time
dt = 0.00545

#Calculate SEM for means DnaK at 37C

root = tk.Tk()
root.withdraw()

#Set working directory, files will be saved here
folder_path = filedialog.askdirectory()
os.chdir(folder_path)

#Open files
file_path = filedialog.askopenfilenames(title="Select CSV files", filetypes=[("CSV files", "*.csv")])

dataframes = []
for file in file_path:
    dataframes.append(pd.read_csv(file))


df_off = []
#Get last 60s of each dataframe
for df in dataframes:
    df["time"]= df["frame"]*dt
    time_course_length = max(df["time"])
    last_60_s = round(time_course_length - 60)
    df = df[df["time"]> last_60_s]
    df_off.append(df)

#Aggregate dataframes

df_agg = []
for df in df_off:
    agg_sem = df.groupby(
        ['mask_uuid', 'cell_id_int', 'particle']).agg({
        'Dapp_log': 'mean',
        'Dapp': 'mean',
        'comment': 'first',
        "track_length": "first"})
    df_agg.append(agg_sem)


 
means = []

for data in df_agg:
    m = data['Dapp_log'].mean()
    print(m)
    means.append(m)
    
sem = stats.sem(means)
print(sem)



#Process and plot the full experiment
root = tk.Tk()
root.withdraw()

#Set working directory, files will be saved here
folder_path = filedialog.askdirectory()
os.chdir(folder_path)



file_path = filedialog.askopenfilename()



df_full = pd.read_csv(file_path)
df_full["time"]= df_full["frame"]*dt





#Get last 60s of the time course
time_course_length = max(df_full["time"])
last_60_s = round(time_course_length - 60)
df_full = df_full[df_full["time"]> last_60_s]



agg_full = []

agg_full = df_full.groupby(
    ['mask_uuid', 'cell_id_int', 'particle']).agg({
    'Dapp_log': 'mean',
    'Dapp': 'mean',
    'comment': 'first',
    "track_length": "first"})


#Last 60s of all acquisitions (full experiment with all replicates)


# Create a dummy x so  all points are placed in one "category"
agg_plot = agg_full.copy()
agg_plot['group'] = 'all'

# Plot the jitter/strip
fig, ax = plt.subplots(figsize=(6, 6))
sns.stripplot(data=agg_plot, x='group', y='Dapp_log', jitter=0.30, size=4, alpha=0.5, ax=ax)

# Compute mean and spread
y = agg_plot['Dapp_log'].to_numpy()

all_D_values = []
all_D_values.extend(agg_plot["Dapp_log"].dropna().to_numpy())
# Convert all Dapp values to numpy array
all_D_values = np.array(all_D_values, dtype=float)

# MATLAB-equivalent mean:
# This matches mean(D) in MATLAB after concatenating all A.Dapp values
mu = np.mean(all_D_values)


# Choose one:
use_sem = True  # set True if you want standard error of the mean
if use_sem:
    spread = sem  # SEM for last 60s of DnaK 37C 3 replicates
    band_label = 'Mean ± 1 SEM'
else:
    spread = np.std(y, ddof=1)  # SD
    band_label = 'Mean ± 1 SD'

# Draw the mean line
ax.axhline(mu, color='crimson', lw=2, zorder=3, label='Mean')

# Shade the band using fill_between across the x-limits
x = np.linspace(*ax.get_xlim(), 200)
ax.fill_between(x, mu - spread, mu + spread, color='crimson', zorder=2, label=band_label)

# Cosmetics
ax.set_xlabel('')
ax.set_ylabel('Dapp_log')
ax.set_title('Dapp_log with jitter (stripplot)')
ax.set_xticks([])

# Optional legend
ax.legend(frameon=False, loc='upper right')
plt.ylim(-2.0, 1.5)

plt.tight_layout()
plt.show()





"""
This section plots the scatterplot  of DnaK's Dapp_log at onset of heat shock with SEM
Use one file with all three replicates of tracking data 

"""
#Calculating SEM values
file_path = filedialog.askopenfilename()
df_hs = pd.read_csv(file_path)

dt = 0.00545
# Time and binning
df_hs["time"] = df_hs["frame"] * dt

agg_hs = df_hs.groupby(
    ['mask_uuid', 'cell_id_int', 'particle']).agg({
    'Dapp_log': 'mean',
    'Dapp': 'mean',
    'comment': 'first',
    "time": "mean"})



agg_hs.sort_values(by="time", inplace=True)
agg_hs_plot = agg_hs.copy()


max_seconds = np.max(agg_hs["time"])
bin_edges = np.arange(0, max_seconds + 1, 5)

# Bin centers for plotting
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

n_bins = len(bin_edges) - 1
all_bins = np.arange(n_bins)
#Bin mean SEMs from three experiments

# Assign global bin index to each row using the same bin_edges
bin_idx = np.digitize(agg_hs["time"].values, bins=bin_edges) - 1
valid = (bin_idx >= 0) & (bin_idx < n_bins)
agg_hs = agg_hs.loc[valid].copy()
agg_hs["bin"] = bin_idx[valid]

per_exp_stats = (
    agg_hs.groupby(["comment", "bin"])["Dapp_log"]
       .agg(mean="mean", sem=sem_func, count="size")
       .reset_index()
)


# Pivot to wide and back to align bins cleanly
per_exp_mean = (
    per_exp_stats.pivot(index="bin", columns="comment", values="mean")
                 .reindex(all_bins)
)

across_mean = per_exp_mean.mean(axis=1)
exp_counts = per_exp_mean.count(axis=1)
across_std = per_exp_mean.std(axis=1, ddof=1)
across_sem = across_std / np.sqrt(exp_counts)




#Plotting 

print("The mean Dapp at heat shock is:", np.mean(agg_hs_plot["Dapp"]))


max_seconds = np.max(agg_hs_plot["time"])
bin_edges = np.arange(0, max_seconds + 1, 5)

# Bin centers for plotting
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2




# Bin centers for plotting
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2


plt.plot(bin_centers, across_mean,label= "Binned arverage", marker='o')
plt.fill_between(bin_centers, across_mean - across_sem, across_mean + across_sem, color='blue', label='±SEM')

plt.scatter(
    agg_hs_plot["time"], agg_hs_plot["Dapp_log"],
    s = 12,
    color = "r"
)

plt.xlim(0,60)
plt.ylim(-2.0, 1.5)
plt.xlabel("Time (s)")
plt.ylabel("Dapp_log")
plt.legend()
plt.tight_layout()
plt.show()









