# -*- coding: utf-8 -*-


import extrack
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from histograms import len_hist



root = tk.Tk()
root.withdraw()

#Set working directory, files will be saved here
folder_path = filedialog.askdirectory()
os.chdir(folder_path)




dt = 0.00545
#ExTrack
#Choose preprocessed file

file_paths = filedialog.askopenfilenames(title="Select CSV files", filetypes=[("CSV files", "*.csv")])

#List of dictionaries containing the fixed Dapp values for each state per time point eg., {'D0': 0.1487 , 'D1': 2.5216, 'D2': 7.7711}

ls_dc = []

vary_params = {'LocErr' : True, 'D0' : False, 'D1' : False, 'D2' : False, 'F0' : True, 'F1' : True, "F2": True, 'p01' : True, 'p02' : True, 'p10' : True, 'p12' : True, 'p20' : True, 'p21' : True, 'pBL' : False}
estimated_vals =  {'LocErr' : 0.02, 'D0' : 0.125, 'D1' : 2.201, 'D2' : 4.638, 'F0' : 0.54, 'F1' : 0.32, "F2":0.14 ,'p01' : 0.05, 'p02' : 0.05, 'p10' : 0.05, 'p12' : 0.05, 'p20' : 0.05, 'p21' : 0.05, 'pBL' : 0.001}
min_values = {'LocErr' : 0.007, 'D0' : 0.125, 'D1' : 2.201, 'D2' : 4.638, 'F0' : 0.001, 'F1' :0.001, "F2" : 0.001, 'p01' : 0.00005, 'p02' : 0.00005, 'p10' : 0.00005, 'p12' : 0.00005, 'p20' : 0.00005, 'p21' : 0.00005, 'pBL' : 0.001}
max_values = {'LocErr' : 0.6, 'D0' :0.125, 'D1' : 2.201, 'D2' : 4.638, 'F0' : 0.9, 'F1' :0.9, "F2": 0.9, 'p01' : 1., 'p02' : 1., 'p10' : 1., 'p12' : 1., 'p20' : 1., 'p21' : 1., 'pBL' : 0.99}


for file, dvals in zip(file_paths, ls_dc):
    #Read file
    all_tracks, frames, opt_metrics = extrack.readers.read_table(file, # path of the file to read or list of paths.
                                          lengths = np.arange(10,300), # number of positions per track accepted (take the first position if longer than max.
                                          dist_th = 0.6, # maximum distance allowed for consecutive positions. 
                                          frames_boundaries = [0, 20000], # minimum and maximum frames allowed.
                                          fmt = 'csv', # format of the document to be red, 'csv' or 'pkl', one can also just specify a separator e.g. ' '. 
                                          colnames = ['POSITION_X', "POSITION_Y", 'FRAME', 'TRACK_ID'],
                                          remove_no_disp = False)

    est_vals = estimated_vals.copy()
    est_vals.update(dvals)
    #print(est_vals)
    min_vals= min_values.copy()
    min_vals.update(dvals)
    #print(min_vals)
    max_vals = max_values.copy() 
    max_vals.update(dvals)
    #print(max_vals)
    
    print('D2:', est_vals.get('D2'), min_vals.get('D2'), max_vals.get('D2'), vary_params.get('D2'), vary_params.get('F2'))

    
    params = extrack.tracking.get_params(nb_states = 3,
                        steady_state = False,
                        vary_params = vary_params,
                        estimated_vals = est_vals,
                        min_values = min_vals,
                        max_values = max_vals)
    #######################################
    target_D2 = float(est_vals.get('D2'))
    current_D0 = params['D0'].value
    current_D1_minus_D0 = params['D1_minus_D0'].value
    d2_minus_d1_val = target_D2 - (current_D0 + current_D1_minus_D0)
    params['D2_minus_D1'].set(value=d2_minus_d1_val, vary=False)
    
    #  enforce resolvable dwell in state 2 with leave+split parameterization ---
    # Idea:
    #   - s2leave = total per-frame probability to leave S2 (p20 + p21)
    #   - f2to0   = fraction of those leaves that go to S0
    #   Then:
    #   - p20 = s2leave * f2to0
    #   - p21 = s2leave * (1 - f2to0)
    #
    # Bounds:
    #   - We cap s2leave from above to ensure a minimum resolvable dwell of Nmin frames.
    #     For Nmin frames, s2leave_max = 1 - exp(-1 / Nmin) (continuous-time-consistent).
    #   - We use a tiny epsilon > 0 as the lower bound to avoid exact zeros (numerical stability).
    #   - f2to0 is a split in [0, 1].
    
    Nmin = 3.0              # desired minimum resolvable dwell in frames (choose 3–5)
    eps = 1e-8              # tiny lower bound to avoid exact zero (effectively zero)
    s2leave_max = 1.0 - np.exp(-1.0 / Nmin)  # ≈ 0.2835 for Nmin=3 #Cap on “how fast you’re allowed to leave state 2 per frame” so that dwells are not shorter than about Nmin frames.
    
    # Initialize from current p20/p21 values (if any); stay within [eps, s2leave_max]
    p20_init = max(params['p20'].value, 0.0)
    p21_init = max(params['p21'].value, 0.0)
    sum_init = p20_init + p21_init
    
    if sum_init <= 0.0:
        # If both are zero, start with a modest interior value and a neutral split
        s2leave_init = max(eps * 10.0, min(0.05, s2leave_max * 0.5))
        f2to0_init = 0.5
    else:
        # Otherwise, start near the current total leave but below the max cap
        s2leave_init = np.clip(sum_init, eps * 10.0, s2leave_max * 0.9)
        f2to0_init = np.clip(p20_init / sum_init, 0.0, 1.0)
    
    # Add or update the two new parameters
    if 's2leave' in params:
        params['s2leave'].set(value=s2leave_init, min=eps, max=s2leave_max, vary=True)
    else:
        params.add('s2leave', value=s2leave_init, min=eps, max=s2leave_max, vary=True)
    
    if 'f2to0' in params:
        params['f2to0'].set(value=f2to0_init, min=0.0, max=1.0, vary=True)
    else:
        params.add('f2to0', value=f2to0_init, min=0.0, max=1.0, vary=True)
    
    # Tie p20 and p21 to s2leave and f2to0, and stop varying them directly.
    # Note: This overrides vary_params entries for p20/p21 (kept unchanged for appearance).
    params['p20'].set(expr='s2leave * f2to0',       vary=False)
    params['p21'].set(expr='s2leave * (1 - f2to0)', vary=False)
    
    # -------------------------------------------------------------


    
    
        
    ######################
    print(params)
    # Parameter fitting:
    model_fit = extrack.tracking.param_fitting(all_tracks = all_tracks,
                                              dt = dt,
                                              params = params,
                                              nb_states = 3, #2 or 3
                                              nb_substeps = 1,
                                              cell_dims = [1],
                                              frame_len = 4,
                                              verbose = 0,
                                              workers = 1, # increase the number of CPU workers for faster computing, do not work on windows or mac (keep to 1).
                                              steady_state = False,
                                              input_LocErr = None
                                              )
    
    params = model_fit.params
    for param in params:
        print(param, params[param].value)
    
    
        
        
    ###############
    #Get state histograms
    
    
    seg_len_hists = len_hist(all_tracks,
                 params=params, 
                 dt=dt, 
                 cell_dims=[1], 
                 nb_states=3, 
                 max_nb_states = 200,
                 workers = 1,
                 nb_substeps=1,
                 input_LocErr = None
                 )
    
    
    
    #############################
    params = model_fit.params
    df = pd.DataFrame(
        [(name, p.value) for name, p in params.items()],
        columns=['param', 'value']
    )
    df.to_csv('params_filename_here.csv', encoding='utf-8', index=False)

    
    ######################################
    #Save seg_len_hists for plotting later also
    
    hists = pd.DataFrame(seg_len_hists)
    
    hists.to_csv("filename_here_hists.csv", encoding='utf-8', index=False)

###############

