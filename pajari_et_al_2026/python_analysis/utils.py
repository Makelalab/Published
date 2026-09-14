# -*- coding: utf-8 -*-

from datetime import time
import h5py
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os
from decimal import Decimal, ROUND_HALF_UP


##################################################################################################################################

#Aggregates tracking data
def aggregate_tracks():
    """
    Script takes csv-files resulting from the tracking pipeline (trackpy) 
    and aggregates single-molecule tracks by particle
    Saves the files in the working directory.
    """
    #Open files
    root = tk.Tk()
    root.withdraw()
    
    #Set working directory, files will be saved here
    folder_path = filedialog.askdirectory()
    os.chdir(folder_path)
    
    #Plotting Multiple Python Pipeline Experiments
    file_path = filedialog.askopenfilenames(title="Select CSV files", filetypes=[("CSV files", "*.csv")])
    
    dataframes = []
    for file in file_path:
        dataframes.append(pd.read_csv(file))
        
        
        
        
    ls = []
    for data in dataframes:
        agg = data.groupby(
            ['mask_uuid', 'cell_id_int', 'particle']).agg({
            'Dapp_log': 'mean',
            'Dapp': 'mean',
            'comment': 'first',
            "track_length": "first"})
        agg.to_csv(f"{data['comment'].iloc[0]}_aggregated.csv")
        ls.append(agg)


def concat_files():
    """
    Script concatenates files of single-molecule tracking data. 
    Used to concatenate e.g. triplicates of an experiment in one file.
    Saves the file in the working directory.
    
    """

    #Open files
    root = tk.Tk()
    root.withdraw()
    
    #Set working directory, files will be saved here
    folder_path = filedialog.askdirectory()
    os.chdir(folder_path)
    
    #Open files 
    
    file_path = filedialog.askopenfilenames(title="Select CSV files", filetypes=[("CSV files", "*.csv")])
    
    #Read all experiments into a list of dataframes
    dataframes = []
    for file in file_path:
        dataframes.append(pd.read_csv(file))
        
    
    combined = pd.DataFrame()
    for file in dataframes:
        combined = pd.concat([combined, file], axis =0)
    filename= input("Type filename here:")

            
    combined.to_csv(f"{filename}.csv", encoding='utf-8', index=False)
    

def separate_files():
    """
    Separate individual fields-of-views from the whole experiment file resulting from the SMT analysis pipeline (trackpy)
    Saves the files in the working directory.
    """

    root = tk.Tk()
    root.withdraw()
    
    #Set working directory, files will be saved here
    folder_path = filedialog.askdirectory()
    os.chdir(folder_path)
    
    #Open files to separate
    
    file_path = filedialog.askopenfilename()
      
    D_cof= pd.read_csv(file_path)
    D_cof['Group'] = D_cof['comment'].str[-4]

    #Returns a tuple containing the group as key and a Dataframe 
    df = sorted(D_cof.groupby('Group'))
    
    for key, data in df:
        data= data.drop('Group', axis=1)
        data.to_csv(f"{data['comment'].iloc[0]}_separate.csv")

def convert_zero_day_timedelta_to_time(td):
    if td.days == 0:
        minute = (td.seconds // 60) % 60
        second = td.seconds % 60
        t_point = str(time(minute=minute, second=second))
        times = t_point[3:]
        return times
    return None



def convert_time_to_offset(time_str, reference_total_seconds):
    minutes, seconds = map(int, time_str.split(':'))
    total_seconds = minutes * 60 + seconds
    return total_seconds - reference_total_seconds


def format_offset_to_time(offset_seconds):
    minutes = offset_seconds // 60
    seconds = offset_seconds % 60
    return f'{minutes}:{seconds:02d}'



def round_half_up(x):
    return int(Decimal(float(x)).quantize(0, rounding=ROUND_HALF_UP))



def replace(cat):
    return (cat.replace("n_cells_", "")
               .replace("refs", " refs")
               .replace("ge3", "≥3 refs"))


def add_reference_columns(df_main, df_ref):

    # Check lengths
    if len(df_main) != len(df_ref):
        raise ValueError(
            f"Dataframes have different lengths: "
            f"{len(df_main)} vs {len(df_ref)}"
        )

    # Check required columns
    required_cols = [
        "n_cells_without_aggregates",
        "n_cells_total"
    ]

    missing = [col for col in required_cols if col not in df_ref.columns]

    if missing:
        raise KeyError(
            f"Missing required columns in reference dataframe: {missing}"
        )

    # Create a copy so original isn't modified
    result = df_main.copy()

    result["n_cells_0ref"] = df_ref["n_cells_without_aggregates"]
    result["total_cells"] = df_ref["n_cells_total"]

    return result

def preprocess_offest(filename):
    """
    filename: str
    #---script ensures the frame number starts from 0. Takes tracking
    files (each replicate separately).
    """
    root = tk.Tk()
    root.withdraw()
    
    #Set working directory, files will be saved here
    folder_path = filedialog.askdirectory()
    os.chdir(folder_path)
    
    file_path = filedialog.askopenfilename()
    
    
    df = pd.read_csv(file_path)
    
    offset = min(df["frame"])
        
    df["frame"]= df["frame"]-offset
    
    print(min(df["frame"]))
        
    df.to_csv(f"{filename}", encoding='utf-8', index=False)



# SEM per bin (uses sample std, ddof=1)
def sem_func(x):
    n = len(x)
    if n <= 1:
        return np.nan
    return np.std(x, ddof=1) / np.sqrt(n)
