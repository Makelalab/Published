# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:14:42 2026

@author: pajaria3
"""


import tkinter as tk
from tkinter import filedialog
import os
import pandas as pd
import numpy as np

"""
Script to calculate the median distances to aggregates for the FumA/MetE-GFP overexpression strains
Requires csv files containing all three experiments per strain analyzed with histograms_distance_to_aggregates.py
(required column= [nearest_ref_dist_nm]).

"""


# GUI file selection
root = tk.Tk()
root.withdraw()

# Set working directory, files will be saved here
folder_path = filedialog.askdirectory(title="Select output folder")
if folder_path:
    os.chdir(folder_path)

# Pick CSVs to plot as separate stacked bars
file_paths = filedialog.askopenfilenames(
    title="Select CSV files",
    filetypes=[("CSV files", "*.csv")]
)

# Read CSVs
dataframes = [pd.read_csv(fp) for fp in file_paths]


#Calculate the median distance to aggregate + SEM per file
# Containers: lists per category
exp_medians_by_cat        = {"slow": [], "trans": [], "fast": []}
sem_of_medians_by_cat     = {"slow": [], "trans": [], "fast": []}
median_of_medians_by_cat  = {"slow": [], "trans": [], "fast": []}  

for idx, data in enumerate(dataframes):
    label = f"Dataframe {idx+1}"  
    
    masks = {
        "slow":  (data["Dapp_log"] <= -0.34),
        "trans": (data["Dapp_log"] >= -0.34) & (data["Dapp_log"] <= 0.74),
        "fast":  (data["Dapp_log"] >= 0.74),
    }

    per_cat_medians = {}
    per_cat_sems    = {}
    per_cat_median_of_medians = {}

    for cat, m in masks.items():
        subset = data.loc[m, ["experiment", "nearest_ref_dist_nm"]].dropna()
        # one median per experiment
        exp_medians = subset.groupby("experiment")["nearest_ref_dist_nm"].median()
        # SEM across those per-experiment medians
        sem_of_medians = exp_medians.sem()
        if pd.isna(sem_of_medians):
            sem_of_medians = 0.0

        # Median across experiments (median of the per-experiment medians)
        median_of_medians = float(exp_medians.median()) if len(exp_medians) else np.nan

        # Save for later
        exp_medians_by_cat[cat].append(exp_medians)
        sem_of_medians_by_cat[cat].append(sem_of_medians)
        median_of_medians_by_cat[cat].append(median_of_medians)

        # For printing
        per_cat_medians[cat] = exp_medians
        per_cat_sems[cat] = sem_of_medians
        per_cat_median_of_medians[cat] = median_of_medians

    # Overall median across categories 
    overall_median_across_cats = np.nanmedian(
        [per_cat_median_of_medians[c] for c in ["slow", "trans", "fast"]]
    )

    # Print once per dataframe
    print(f"\n{label}:")
    for cat in ["slow", "trans", "fast"]:
        emed = per_cat_medians[cat]
        se   = per_cat_sems[cat]
        momd = per_cat_median_of_medians[cat]
        print(f"  {cat}: n_experiments={len(emed)}, median_of_medians={momd:.3f} nm, SEM(of medians)={se:.3f} nm")
        if len(emed):
            print("    per-experiment medians (nm):")
            print("    " + emed.round(3).to_string().replace("\n", "\n    "))
        else:
            print("    per-experiment medians (nm): none")

    print(f"  overall median across categories (median of category medians): {overall_median_across_cats:.3f} nm")
    
    
##For all dataframes in one 
    
 
# 1) Concatenate all dataframes (treat as a single dataset)
df_all = pd.concat(dataframes, ignore_index=True)

# 2) Define categories on the combined dataframe
masks = {
    "slow":  (df_all["Dapp_log"] <= -0.34),
    "trans": (df_all["Dapp_log"] >= -0.34) & (df_all["Dapp_log"] <= 0.74),
    "fast":  (df_all["Dapp_log"] >= 0.74),
}

summary_rows = []
per_exp_medians = {}  # category -> Series (index=experiment, values=median)

for cat, mask in masks.items():
    subset = df_all.loc[mask, ["experiment", "nearest_ref_dist_nm"]].dropna()
    # Per-experiment medians on the combined data
    exp_medians = subset.groupby("experiment")["nearest_ref_dist_nm"].median()
    per_exp_medians[cat] = exp_medians

    n_experiments = int(exp_medians.size)
    sem_of_medians = exp_medians.sem()
    if pd.isna(sem_of_medians):
        sem_of_medians = 0.0
    median_of_medians = float(exp_medians.median()) if n_experiments else np.nan

    summary_rows.append({
        "category": cat,
        "n_experiments": n_experiments,
        "median_of_medians_nm": median_of_medians,
        "sem_of_medians_nm": sem_of_medians,
    })

# 3) Tidy summary
summary_df = pd.DataFrame(summary_rows).sort_values("category").reset_index(drop=True)

# 4) Print summary and the per-experiment medians
print("\nSummary on concatenated data (treated as one dataset):")
for _, r in summary_df.iterrows():
    print(
        f"- {r['category']}: n_experiments={int(r['n_experiments'])}, "
        f"median_of_medians={r['median_of_medians_nm']:.3f} nm, "
        f"SEM(of medians)={r['sem_of_medians_nm']:.3f} nm"
    )

print("\nPer-experiment medians (nm) by category:")
for cat in ["slow", "trans", "fast"]:
    s = per_exp_medians.get(cat, pd.Series(dtype=float))
    if s.empty:
        print(f"- {cat}: none")
    else:
        print(f"- {cat}:")
        print("  " + s.round(3).to_string().replace("\n", "\n  "))

