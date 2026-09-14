# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 18:36:03 2026

@author: pajaria3
"""

import tkinter as tk
from tkinter import filedialog
import os
import pandas as pd
import numpy as np
from scipy import stats

"""
This script calculates and outputs the number of tracks and cells in
a csv-file (output file from tracking pipeline).
"""

root = tk.Tk()
root.withdraw()

#Set working directory, files will be saved here
folder_path = filedialog.askdirectory()
os.chdir(folder_path)

#Open files
file_path = filedialog.askopenfilenames(
    title="Select CSV files",
    filetypes=[("CSV files", "*.csv")]
)

# Read all experiments into a list of dataframes
dataframes = []

required_cols = [
    "mask_uuid",
    "cell_id_int",
    "particle",
    "Dapp_log",
    "Dapp",
    "comment",
    "track_length"
]

for file in file_path:
    data = pd.read_csv(file)

    # Check required columns are present
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(
            f"File {os.path.basename(file)} is missing required columns: {missing_cols}"
        )

    dataframes.append(data)


all_D_values = []
ls = []
tracks = 0
cell = 0

# store cell counts per file and per mask_uuid
cell_count_tables = []

for file, data in zip(file_path, dataframes):
    filename = os.path.basename(file)
    
    # MATLAB-style D calculation:
    # collect all raw Dapp values directly from each file
    all_D_values.extend(data["Dapp"].dropna().to_numpy())

    # Aggregate tracks using mask_uuid + cell_id_int + particle
    agg = data.groupby(
        ["mask_uuid", "cell_id_int", "particle"],
        as_index=False
    ).agg({
        "Dapp_log": "mean",
        "Dapp": "mean",
        "comment": "first",
        "track_length": "first"
    })

    track_l = len(agg)
    tracks += track_l
    ls.append(agg)

    # Count unique cells as unique mask_uuid + cell_id_int pairs
    n_cells_file = data[["mask_uuid", "cell_id_int"]].drop_duplicates().shape[1]
    cell += n_cells_file

    # Count unique cells per mask_uuid within this file
    cells_per_mask = (
        data.groupby("mask_uuid")["cell_id_int"]
            .nunique()
            .reset_index(name="n_unique_cells")
    )
    cells_per_mask.insert(0, "file", filename)
    cell_count_tables.append(cells_per_mask)



# Combine all raw data
all_data = pd.concat(dataframes, ignore_index=True)

# Combine all aggregated track tables if needed
all_tracks = pd.concat(ls, ignore_index=True)

# Combine per-file/per-mask cell count tables
cell_counts_summary = pd.concat(cell_count_tables, ignore_index=True)

# Convert all Dapp values to numpy array
all_D_values = np.array(all_D_values, dtype=float)

# MATLAB-equivalent mean:
# This matches mean(D) in MATLAB after concatenating all A.Dapp values
D_m = np.mean(all_D_values)


# SEM calculation: between experiments/comments
# -------------------------------------------------------------------------
# First calculate mean Dapp per experiment/comment.
# Then calculate SEM from those experiment means.
exp_means = []

for file, data in zip(file_path, dataframes):
    Dapp_values = pd.to_numeric(data["Dapp"], errors="coerce").dropna()

    mean_Dapp_file = np.mean(Dapp_values)
    exp_means.append(mean_Dapp_file)

    print(f"{os.path.basename(file)} mean Dapp = {mean_Dapp_file}")

sem_between_experiments = stats.sem(
    exp_means,
    nan_policy="omit"
)



# Overall mean of the file means
D_m_between_files = np.mean(exp_means)

print(f"Mean of file means = {D_m_between_files}")
print(f"SEM between file means = {sem_between_experiments}")
print(f"Number of files/experiments used for SEM = {len(exp_means)}")

print("\n==============================")
print("Overall summary")
print("==============================")
print(f"Overall mean Dapp, MATLAB-style from all individual Dapp values: {D_m}")


print(f"Tracks {tracks}")

print("\nCell counts per file and mask_uuid:")


print("Total number of cells:", np.sum(cell_counts_summary["n_unique_cells"]))
