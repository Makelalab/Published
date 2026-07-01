# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 16:38:29 2026

@author: pajaria3
"""

import numpy as np
import h5py
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
from datetime import timedelta
import pandas as pd
import glob
import os
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt
from utils import convert_zero_day_timedelta_to_time
from utils import convert_time_to_offset
from utils import format_offset_to_time


"""
This script plots the VAHEAT heat shock ramp. Requires all h5 files of one experiment to get the 
full VAHEAT data. Default reference values in code for tracking files of experiment 251029_AP_Ec_05_M9glyCAAT_JFX554
"""

###############################################################################
#Get start and end times for each fov in experiment



#Choose folder
root = tk.Tk()
root.withdraw()

folder_path = filedialog.askdirectory()
os.chdir(folder_path)

#Reads all h5-files in folder
file_paths = glob.glob(os.path.join(folder_path, "*.h5"))

df_fovs = pd.DataFrame(columns=["start_dt","end_dt"])
rows = 0

#Get acquisition start and end times from metadata
for file in file_paths:
    f1 = h5py.File(file,'r+')
    start = f1['MetaData'].attrs['StartTime']
    end = f1['MetaData'].attrs['EndTime']
    
    print(f"File: {file}, Start: {start}, End: {end}")
    
    start_dt = datetime.fromtimestamp(start)

    end_dt = datetime.fromtimestamp(end)

    df_fovs.loc[rows] = [start_dt, end_dt]
    rows += 1
    f1.close()
    
#normalize start and end times and convert to minutes and seconds    

df_fovs["start_norm"] = df_fovs["start_dt"]-df_fovs["start_dt"][0]
df_fovs["end_norm"] = df_fovs["end_dt"]-df_fovs["start_dt"][0]

df_fovs['start_str'] = df_fovs['start_norm'].apply(convert_zero_day_timedelta_to_time)
df_fovs['end_str'] = df_fovs['end_norm'].apply(convert_zero_day_timedelta_to_time)



start_ls = df_fovs['start_str'].tolist()
end_ls = df_fovs['end_str'].tolist()
for i in end_ls:
    start_ls.append(i)
############################################################################
#Get complete temperature data from the last .h5 of the experiment

root = tk.Tk()
root.withdraw()
path = filedialog.askopenfilename()
f1 = h5py.File(path,'r+') 

#Get VAHEAT temperature values
try:
    t = f1['MetaData']['temperature'].attrs['temperature'].decode('utf-8')
    np_t = np.array([float(value[1:]) for value in t.split('\na') if value.startswith('F')])
except KeyError:
    t = 'no temperature data'
    

#Get start time
start=f1['MetaData'].attrs['StartTime']
#Start time in huma-readable form
start_hr=str(datetime.fromtimestamp(start).time())


end=f1['MetaData'].attrs['EndTime']
#End time in human-readable form
end_hr=str(datetime.fromtimestamp(end).time())

#Get VAHEAT temperature time points convert to human-readable time
vh = f1['MetaData']['temperature'].attrs["start_time"]

vh_hr = (datetime.fromtimestamp(vh).time())
vh_dt = datetime.fromtimestamp(vh)
vh_mr = vh_dt.time().replace(microsecond=0)
vh_str = vh_mr.strftime("%H:%M:%S")



ls = []
ls.append(vh_dt)


for i in range(1, np.size(np_t)):
    vh_dt += timedelta(seconds=1, milliseconds = 5)
    ls.append(vh_dt)

   
#Dataframe with temperature and time points 
temp_time = pd.DataFrame(columns = ["Temperature", "Time"])
temp_time["Time"] = ls
temp_time["Temperature"] = np_t



#Find the time when heat shock and the following field of view starts
hs = temp_time[temp_time["Temperature"]>=45].iloc[0]
print("The heat shock starts at:",hs["Time"],"The FOV acquisition starts at:", df_fovs["start_dt"][3])

#Normalize times so that the graph starts from when the first FOV starts
temp_time["norm"] = temp_time["Time"]-df_fovs["start_dt"][0]



# Apply the conversion function only to timedeltas with 0 days
temp_time['norm_minutes'] = temp_time['norm'].apply(convert_zero_day_timedelta_to_time)


#Plotting

filtered = temp_time.dropna()

#Onset of heat shock
reference_time = "12:20"
reference_minutes, reference_seconds = map(int, reference_time.split(':'))
reference_total_seconds = reference_minutes * 60 + reference_seconds

# Convert norm_minutes to offset from reference time 
filtered['time_offset'] = filtered['norm_minutes'].apply(lambda x: convert_time_to_offset(x, reference_total_seconds))
start_ls_offsets = [convert_time_to_offset(x, reference_total_seconds) for x in start_ls]

temperature_list = []
for offset in start_ls_offsets:
    selected_temp = filtered.loc[filtered['time_offset'] == offset, 'Temperature']
    if not selected_temp.empty:
        temperature_list.append(selected_temp.values[0])
    else:
        temperature_list.append(np.nan)


# Configure inset axis limits and formatting
zoom_start_offset = convert_time_to_offset("05:50", reference_total_seconds)
zoom_end_offset = convert_time_to_offset("19:30", reference_total_seconds)




fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])


ax_zoom = fig.add_subplot(gs[1, 0])

# Zoomed subplot
ax_zoom.plot(filtered['time_offset'], filtered['Temperature'], color='tab:red')
ax_zoom.set_xlim(zoom_start_offset, zoom_end_offset)
ax_zoom.set_ylim(36, 46)
ax_zoom.xaxis.set_major_locator(ticker.MultipleLocator(60))
ax_zoom.xaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, _: format_offset_to_time(int(x)))
)
ax_zoom.yaxis.set_major_locator(ticker.FixedLocator(np.arange(37, 47, 2)))
ax_zoom.set_title('Heat Shock Ramp')

plt.show()
