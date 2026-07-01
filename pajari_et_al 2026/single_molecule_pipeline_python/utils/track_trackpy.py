import os
import matplotlib.pyplot as plt
import numpy as np
import trackpy as tp
import pandas as pd
from skimage import measure
from utils.tiff_writer import show_mask
from utils.align_mask import align_mask
from utils.logging import log_args_and_time


import logging
logging.getLogger('trackpy').setLevel(logging.WARNING)
# logger from trackpy crashing the process

@log_args_and_time
def track_py(locs, proj_sd, mask_file, tiff_uuid, mask_uuid, 
             flip_mask=True, shortest_track=10, align_m='auto', max_disp_px=10, allow_missing=3, tp_link_strategy=None):

    # getting mask
    directory = os.getenv('SMPP_DATA_DIR')+'/analysis/masks_omni/'
    file = mask_file.split('/')[-1]
    print(file)
    m = show_mask(directory, file, imshow=False)

    # regiter mask and fluorescence signal (SD projection)
    if align_m=='auto':
        transform = align_mask(proj_sd, m, tform='RIGID_BODY', verbose=False) # tform = 'TRANSLATION' 
        # transform = np.linalg.inv(transform) # to transform locs x and y, not the mask 
        # locs['x_t'] = locs['x']*transform[0,0]+locs['y']*transform[0,1]+transform[0,2]
        # locs['y_t'] = locs['x']*transform[1,0]+locs['y']*transform[1,1]+transform[1,2]
        locs['x_t'] = locs['y']*transform[1,0]+locs['x']*transform[1,1]+transform[1,2]
        locs['y_t'] = locs['y']*transform[0,0]+locs['x']*transform[0,1]+transform[0,2]

    elif align_m=='manual':
        pass
        
    else:
        locs['x_t'] = locs['x'] 
        locs['y_t'] = locs['y'] 

    if flip_mask:
        # flip mask to match localizations indexing
        m = np.flip(m, axis=1)


    ttl = 'locs after registration'
    locs_preview_contours(locs['y_t'], locs['x_t'], locs['frame'], m, ttl)
    
    ttl = 'locs before registration'
    locs_preview_contours(locs['y'], locs['x'], locs['frame'], m, ttl)


    # iterate through masks
    all_masks = []
    locs_counter = 0
    tracks_counter = 0
    
    for i in range(1, np.max(m)+1):
        cm = (m==i)
        
        # get locs for mask
        in_cell=locs.copy()
        in_cell['x_rounded']=in_cell['x_t'].astype('int') # no .round(0) here! 
        in_cell['y_rounded']=in_cell['y_t'].astype('int')
        in_cell = in_cell.loc[((in_cell['x_rounded']<cm.shape[0])&(0<in_cell['x_rounded']))&\
                              ((in_cell['y_rounded']<cm.shape[1])&(0<in_cell['y_rounded']))]
        in_cell['in_cell']=cm[in_cell['x_rounded'], in_cell['y_rounded']]
            
        
        in_cell = in_cell.loc[in_cell['in_cell']==True]
        locs_counter += in_cell.shape[0]
        
        # link the xy into tracks
        if in_cell.shape[0]>0:
            t = tp.link(in_cell, max_disp_px, pos_columns=['y_t', 'x_t'], t_column='frame', memory=allow_missing, link_strategy=tp_link_strategy)  
            t1 = tp.filter_stubs(t, shortest_track)
            t1['cell_id_int']=i
            tracks_counter += len(t1['particle'].unique())
            all_masks.append(t1)
            
    
    concatenated = pd.concat(all_masks)
    concatenated['fov_mask_id'] = file
    concatenated['tiff_uuid'] = tiff_uuid
    concatenated['mask_uuid'] = mask_uuid

    # print('filtered locs, color by particle')
    # plt.scatter(concatenated['y_t'], concatenated['x_t'], c=concatenated['particle'], s=1, alpha=0.1)
    # plt.axis('scaled')
    # plt.show()

    print(f'locs in tracks before tracking {locs_counter}')
    print(f'locs in tracks after tracking {concatenated.shape[0]}')
    print(f'tracks found in FOV {tracks_counter}')    
    return concatenated 

def plot_tracks(df):
    cell_mask_idx = df.copy()
    cell_mask_idx.set_index(['fov_mask_id', 'cell_id_int'], inplace = True)
    
    # cell_mask_idx.index.unique()
    for i in cell_mask_idx.index.unique():
        # print(cell_mask_idx.loc[i].shape)
        t1 = cell_mask_idx.loc[i].sort_values(by=['frame'])
         # plotting all tracks in the mask
        for k in t1['particle'].unique():
            track = t1.where(t1['particle']==k).dropna(how='all')
            plt.plot(track['y_t'], track['x_t'])
        plt.axis('scaled')
        # plt.show()
    plt.show()    


def locs_preview_contours(locs_y, locs_x, color, mask, title):
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.scatter(locs_y, locs_x, 
                c=color, 
                s=1, 
                # alpha=0.7, 
                cmap='viridis')
    plt.axis('scaled')
    # plt.show()

    
    
    # fig, ax = plt.subplots()
    # ax.imshow(m, cmap=plt.cm.gray)
    for i in range(np.max(mask)):
        m = (mask==i)
        contours = measure.find_contours(m, 0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=0.5, color='r')
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.title(title)
    plt.show()




