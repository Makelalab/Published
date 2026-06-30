# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from utils import round_half_up
from utils import replace
import tkinter as tk
from tkinter import filedialog
import os
import pandas as pd
import numpy as np
from scipy.stats import sem
from utils import add_reference_columns

#OE intensity control figures
"""
Script to plot the stacked bar graph of the sum of fluorescence intensity in cells in strains expressing
endogenous MetE/FumA-GFP and strains overexpressing MetE/FumA-GFP from
their respective ASKA+ plasmids

Requires the pre-calculated fluorescence intensity sums
 
"""

# Intensity sum labels
labels = ['FumA_intensity_sum', 'MetE_intensity_sum']

# bottom and top segments for each label
bottom = np.array([407274.8815, 357983.0958])  # endogenous expression intensity
top    = np.array([479694.4949, 302349.4002])  # overexpression intensity

# SEMs for each segment (same order)
sem_bottom = np.array([47550, 34250])
sem_top    = np.array([40407, 12519])

x = np.arange(len(labels))
width = 0.5

fig, ax = plt.subplots(figsize=(6,4))

ax.bar(x, bottom, color='cornflowerblue', width=width, yerr=sem_bottom, capsize=4, label='bottom')
ax.bar(x, top, bottom=bottom, color='lavender', width=width, yerr=sem_top, capsize=4, label='top')

ax.set_xticks(x, labels)
ax.set_ylabel('Intensity sum')
ax.set_title('OE intensity control with segment-wise error bars')
plt.tight_layout()
plt.show()


#Calculate SEMs for the fluorescence intensity plot above


root = tk.Tk()
root.withdraw()

#Set working directory, files will be saved here
folder_path = filedialog.askdirectory()
os.chdir(folder_path)


file_path = filedialog.askopenfilenames(title="Select CSV files", filetypes=[("CSV files", "*.csv")])

#Read all experiments into a list of dataframes
dataframes = []
for file in file_path:
    dataframes.append(pd.read_csv(file))
    
#Median fluorescence intensity of wildtype cells (no protein-GFP expression)
wt_median = 605 #Calculated from intensity values from script intensity_in_cells.py

means = []

for data in dataframes:
    data["norm_pxl_sum"] = data["intsum_px"] - wt_median
    exp = data["fov_mask_id"][0]
    sum_int = sum(data["norm_pxl_sum"])
    means.append(sum_int)
    print(f"This is the norm pxl sum for{exp}:", sum_int)
    
means = np.asarray(means, dtype=float)
sem_val = sem(means, ddof=1, nan_policy='omit')

print("N =", np.sum(~np.isnan(means)))
print("SEM =", sem_val)





# --------------------------------------------------
#Preprocessing
# --------------------------------------------------



"""
Add columns ["n_cells_without_aggregates"] and ["n_cells_total"] to file with per cell aggregate data 
and save file
"""
root = tk.Tk()
root.withdraw()


#Set working directory, files will be saved here
folder_path = filedialog.askdirectory()
os.chdir(folder_path)


file_path_exp = filedialog.askopenfilename(title="Select CSV files", filetypes=[("CSV files", "*.csv")])
file_path_ref = filedialog.askopenfilename(title="Select CSV files", filetypes=[("CSV files", "*.csv")])

exp = pd.read_csv(file_path_exp)
ref = pd.read_csv(file_path_ref)
    
        
df_processed = add_reference_columns(exp, ref)
df_processed.to_csv("filename")


# --------------------------------------------------
# Main
# --------------------------------------------------

"""
Plots stacked bar graphs of the aggregate per cell fractions for each strain
Requires columns: "0 refs", "1 ref", "2 refs", "≥3 refs"
"""
root = tk.Tk()
root.withdraw()

experiment_names = [
    "FumA OE",
    "FumA Control",
    "MetE OE",
    "MetE Control"
]

means_list = []
sems_list = []
totals_list = []

for exp_name in experiment_names:

    dataframes = load_experiment(exp_name)

    means, sems, total_cells = get_ref_stats(dataframes)

    means_list.append(means)
    sems_list.append(sems)
    totals_list.append(total_cells)

    print(f"\n{exp_name}")

    for label, mean_val, sem_val in zip(
        ["0 refs", "1 ref", "2 refs", "≥3 refs"],
        means,
        sems
    ):
        print(
            f"{label}: "
            f"mean={mean_val:.4f}, "
            f"SEM={sem_val:.4f}"
        )

# Convert to arrays for plotting
means = np.vstack(means_list)
sems = np.vstack(sems_list)

# --------------------------------------------------
# Plot
# --------------------------------------------------

categories = [
    "0 refs",
    "1 ref",
    "2 refs",
    "≥3 refs"
]

colors = [
    "steelblue",
    "orange",
    "gold",
    "forestgreen"
]

x = np.arange(len(experiment_names))

fig, ax = plt.subplots(figsize=(8, 5))

bottom = np.zeros(len(experiment_names))

for i in range(4):

    ax.bar(
        x,
        means[:, i],
        bottom=bottom,
        color=colors[i],
        yerr=sems[:, i],
        capsize=4,
        label=categories[i]
    )

    bottom += means[:, i]

# Add total cell counts above bars
for i, total in enumerate(totals_list):

    ax.text(
        x[i],
        bottom[i] + 0.02,
        f"Total: {total:,}",
        ha="center",
        fontsize=9
    )

ax.set_xticks(x)
ax.set_xticklabels(experiment_names)

ax.set_ylabel("Mean fraction of cells")
ax.set_title("Reference category distribution")
ax.legend(title="Category")

plt.tight_layout()
plt.show()