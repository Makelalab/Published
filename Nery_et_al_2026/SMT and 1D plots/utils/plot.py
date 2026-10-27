import numpy as np
import pandas as pd
import seaborn as sns
import os
import ipywidgets as widgets

from scipy import stats
from ipywidgets import Button, Layout, interactive, fixed
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import label, find_objects
from skimage import measure
from datetime import datetime
from pathlib import Path
from seaborn.axisgrid import JointGrid

from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from utils.xml_writer import get_exps_meta_from_xml
from utils.logging import setup_logger, log_and_print, save_as_pickle
from utils.get_features import get_area, to_PC_raw




def show_mask(directory, mask_choice, imshow=True):
    
    if mask_choice[-4:] == '.png':
        #print('png')
        m = Image.open(Path(directory)/mask_choice)  
        m = np.array(m)
        shfld = np.arange(10, np.max(m)+10)
        np.random.shuffle(shfld)

        mm = np.where(m == 0, 0, m)
        for i in range(1, np.max(m)):
            mm = np.where(m == i, shfld[i], mm)

        if imshow:
            plt.imshow(mm, cmap = 'Blues')
            plt.show()    
    
    if mask_choice[-4:] == '.npy':
        #print('npy')
        m_arr = np.load(Path(directory)/mask_choice, allow_pickle=True)
        m = m_arr.item()
        m = m['masks']
        shfld = np.arange(10, np.max(m)+10)
        np.random.shuffle(shfld)

        mm = np.where(m == 0, 0, m)
        for i in range(1, np.max(m)):
            mm = np.where(m == i, shfld[i], mm)

        if imshow:
            plt.imshow(mm, cmap = 'Blues')
            plt.show()    
            
    return m

def locs_preview_contours(locs_y, locs_x, color, mask, title, size=(15, 15)):
    fig, ax = plt.subplots(figsize=size)
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


def locs_preview_contours_splots(locs_y, locs_x, color, mask, title,
                         ax=None, size=(15, 15), show=True):
    # Create ax only if caller didn't provide one
    if ax is None:
        fig, ax = plt.subplots(figsize=size)  # size only relevant here [web:2]
    else:
        fig = ax.figure

    sc = ax.scatter(locs_y, locs_x, c=color, s=1, cmap='viridis')
    ax.set_aspect('equal', adjustable='box')

    for i in range(int(np.max(mask))):
        m = (mask == i)
        contours = measure.find_contours(m, 0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=0.5, color='r')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

    if show and ax is not None and fig is not None:
        # If you're in a notebook, this is usually optional; kept to match your pattern.
        plt.show()
        return
    return ax  # always return ax so it can be used in subplots

def locs_preview_contours_sc(locs_y, locs_x, color, mask, title, size=(15, 15), smooth=0):
    from scipy.interpolate import splprep, splev
    fig, ax = plt.subplots(figsize=size)
    plt.scatter(locs_y, locs_x, 
                c=color, 
                s=3, 
                cmap='viridis')
    plt.axis('scaled')

    for i in range(1, np.max(mask) + 1):  # Ensure iteration covers all object indices
        m = (mask == i)
        contours = measure.find_contours(m, 0.5)
        for contour in contours:

            # Smooth the contour using B-spline interpolation
            if len(contour) >= 4:  # Need at least 4 points to perform B-spline interpolation
                x = contour[:, 1]
                y = contour[:, 0]

                # Fit B-spline - Degree of spline (k=3) for cubic spline
                tck, u = splprep([x, y], s=smooth)
                u_new = np.linspace(u.min(), u.max(), len(x) * 2)  # More points for smoothness
                x_new, y_new = splev(u_new, tck)

                ax.plot(x_new, y_new, linewidth=0.5, color='r')
            else:
                # Fallback to non-smoothed plot if not enough points
                ax.plot(contour[:, 1], contour[:, 0], linewidth=0.5, color='r')

        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.title(title)
    plt.show()
 



def annotate_mask(mask, size=(15, 15)):
    mask = np.flipud(mask)

    # Get unique non-zero values in the mask
    unique_values = np.unique(mask)
    unique_non_zero_values = unique_values[unique_values != 0]
    
    # Create colormap with shuffled non-zero colors
    cmap = plt.cm.get_cmap('tab20', len(unique_non_zero_values))
    non_zero_colors = cmap(np.arange(cmap.N))
    np.random.shuffle(non_zero_colors)
    
    # Insert black color for zero value at the beginning
    colors = np.vstack([[0, 0, 0, 1], non_zero_colors])
    
    # Create a custom colormap
    custom_cmap = ListedColormap(colors)
    
    # Prepare to plot the mask
    plt.figure(figsize=size)  # Adjust width and height to suit your needs
    plt.imshow(mask, cmap=custom_cmap, interpolation='none')
    
    # Iterate through each unique value to label each specific region connected with the same value
    for unique_value in np.unique(mask):
        if unique_value == 0:
            continue  # Skip background
    
        # Create a binary mask for the current unique value
        current_value_mask = (mask == unique_value)
    
        # Label the connected components in this binary mask
        labeled_array, num_features = label(current_value_mask)
    
        # Process each connected region for the current value
        for region_num in range(1, num_features + 1):
            # Obtain slice object for the connected component
            slice_x, slice_y = find_objects(labeled_array == region_num)[0]
    
            # Calculate the centroid to place the label
            centroid_x = (slice_x.start + slice_x.stop - 1) / 2
            centroid_y = (slice_y.start + slice_y.stop - 1) / 2
    
            # Place the label on the plot at the centroid
            plt.text(centroid_y, centroid_x, unique_value, ha='center', va='center', color='red')
    
    # plt.colorbar()
    plt.axis('off')
    
    plt.show()

def locs_preview_contours_sc_edges(locs_y, locs_x, color, mask, title, size=(15, 15), smooth=0, output_path=''):
    from scipy.interpolate import splprep, splev
    fig, ax = plt.subplots(figsize=size)
    sns.lineplot(x=locs_y, y=locs_x, 
                 sort=False,
                 # hue=color, 
                 legend=False,
                # s=3, 
                # cmap='viridis',
            )
    plt.axis('scaled')

    for i in range(1, np.max(mask) + 1):  # Ensure iteration covers all object indices
        m = (mask == i)
        contours = measure.find_contours(m, 0.5)
        for contour in contours:

            # Smooth the contour using B-spline interpolation
            if len(contour) >= 4:  # Need at least 4 points to perform B-spline interpolation
                x = contour[:, 1]
                y = contour[:, 0]

                # Fit B-spline - Degree of spline (k=3) for cubic spline
                tck, u = splprep([x, y], s=smooth)
                u_new = np.linspace(u.min(), u.max(), len(x) * 2)  # More points for smoothness
                x_new, y_new = splev(u_new, tck)

                ax.plot(x_new, y_new, linewidth=0.5, color='r')
            else:
                # Fallback to non-smoothed plot if not enough points
                ax.plot(contour[:, 1], contour[:, 0], linewidth=0.5, color='r')

        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.title(title)
    save_path = os.path.join(output_path, f'{datetime.today().strftime("%Y-%m-%d")}_{title}.svg')
    plt.savefig(save_path)
    print(f'image saved in {save_path}')
    plt.show()




def manual_frame_range(fovs_list, frame_range, fov_n, ranges_tuple):
    
    ranges_dict = ranges_tuple[0]
    # all_fovs_locs 
    
    all_fovs_locs = fovs_list
    file = all_fovs_locs[fov_n][2]#.split('/')[-1]
    print(file)
    s_all_tracks = all_fovs_locs[fov_n][0] 

    by_frame = s_all_tracks.groupby('frame').agg({'frame':'count', 
                                            'mass':'median', 
                                            'size_x':'median', 
                                            'size_y':'median',
                                            'ecc':'median',
                                            'signal':'median', 
                                            'raw_mass':'median', 
                                            'ep':'median',
                                            # 'track_length':'median', 
                                            # 'Dapp_log':'median', 
                                            # 'intensity_2ch_by_track':'median',
                                           }).rename(columns={'frame':'locs_in_frame'})

    # Function to plot data with average and error
    def plot_with_average_and_error(ax, data, column, window_size=10):
        # Compute rolling mean and rolling standard deviation
        rolling_mean = data[column].rolling(window=window_size, center=True).mean()
        rolling_std = data[column].rolling(window=window_size, center=True).std()
        
        # Plot on the given axis
        ax.plot(data.index, rolling_mean, label=f'{column} Average')
        ax.fill_between(data.index, rolling_mean - rolling_std, rolling_mean + rolling_std, alpha=0.3)
        ax.set_title(column)
        ax.set_xlim(data.index[0], data.index[-1])  # Limit x-axis
    
    # Subset the DataFrame to the first n frames
    subset_by_frame = by_frame.iloc[frame_range[0]:frame_range[1]]

    # Create a figure with 10 subplots (adjust rows x columns)
    num_plots = len(subset_by_frame.columns)
    cols = 4
    rows = (num_plots + cols - 1) // cols  # Compute number of rows needed
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, 5))
    axes = axes.flatten()  # Flatten the axes array to easily iterate
    
    # Loop through the columns, plot each one in a different subplot
    for i, column in enumerate(subset_by_frame.columns):
        plot_with_average_and_error(axes[i], subset_by_frame, column, window_size=200)
    
    # Hide any unused subplots
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])
    
    # Adjust layout for better visualization
    
    plt.tight_layout()
    plt.show()
  

    
    button = widgets.Button(description="set frame range")
    output = widgets.Output()
    
    display(button, output)
    
    def on_button_clicked(b):
        with output:
            
            # s_all_tracks = s_all_tracks[s_all_tracks['frame']>=frame_range[0]]
            # s_all_tracks = s_all_tracks[s_all_tracks['frame']<=frame_range[1]]
            ranges_dict[file]=frame_range
            print(f'frame range for FOV {file} is set to: {frame_range}')
    
    button.on_click(on_button_clicked)


def ridges_side_by_side(data, output_directory, width=12, as_svg=False):
    def ridge_tracks_vs_locs_save(df, filename, as_svg):
        df = df.reset_index()
        Temp_data = df[["mask_uuid", "comment", "Dapp_log"]].copy()
        Temp_data["Density_coor_id"] = Temp_data["mask_uuid"] + '_' + Temp_data["comment"]
        for i in range(Temp_data.shape[0]):
            Temp_data.loc[i, "Density_coor_id"] = Temp_data.loc[i, "Density_coor_id"][:8] + '-' + Temp_data.loc[i, "Density_coor_id"][-11:-3]
        Plot_data = Temp_data[["Density_coor_id", "Dapp_log"]].copy()
        pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
        
        g = sns.FacetGrid(Plot_data, row="Density_coor_id", hue="Density_coor_id", aspect=4, height=2, palette=pal)
        g.map_dataframe(sns.histplot, x="Dapp_log", kde=True, stat="density", bins=30, kde_kws={"bw_adjust": .5})
        # g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
    
        for ax in g.axes.flat:
            ax.spines['bottom'].set_color('black')
            ax.spines['bottom'].set_linewidth(1)  # normal thickness
        
        def label(x, color, label):
            ax_ = plt.gca()
            ax_.text(.02, .8, label, ha="left", va="top", transform=ax_.transAxes)
    
        g.set(ylim=(0, 1.8))
        g.map(label, "Dapp_log")
        g.figure.subplots_adjust(hspace=0.1)
        g.set_titles("")
        g.despine(bottom=False, left=True)
        
        g.savefig(filename)
        if as_svg:
            save_path = os.path.join(output_directory.selected_path, f"{filename[:-4]}_{datetime.today().strftime('%Y-%m-%d')}.svg")
            plt.savefig(save_path)
            print(f'image saved in {save_path}')

        plt.close(g.fig)
    
    df1 = data
    df2 = data.reset_index().groupby(['mask_uuid', 'cell_id_int', 'particle'], as_index=False).agg({'Dapp_log':'first', 'comment':'first'})

    left_path = os.path.join(output_directory.selected_path, 'ridge_left.png')
    right_path = os.path.join(output_directory.selected_path, 'ridge_right.png')
    
    ridge_tracks_vs_locs_save(df1, left_path, as_svg)
    ridge_tracks_vs_locs_save(df2, right_path, as_svg)
    
    # Now load the saved plots and display side-by-side
    left_img = plt.imread(left_path)
    right_img = plt.imread(right_path)

    # Remove the PNG files
    os.remove(left_path)
    os.remove(right_path)
    
    aspect = left_img.shape[0] / left_img.shape[1]
    w = width 
    
    fig, axs = plt.subplots(1, 2, figsize=(w, w/2*aspect)) #  figsize=()
    axs[0].imshow(left_img)
    axs[0].axis('off')
    axs[0].set_title('Non-aggregated')
    
    axs[1].imshow(right_img)
    axs[1].axis('off')
    axs[1].set_title('Aggregated by particle')
    
    plt.tight_layout()
    plt.show()


def plot_all_locs(data, masks_directory, color = 'Dapp_log'):
    directory = masks_directory
    for i in data['fov_mask_id'].unique():
        m = show_mask(directory, i, imshow=False)
        
        locs_subset = data.loc[data['fov_mask_id']==i]
        locs_y = locs_subset['y_t']
        locs_x = locs_subset['x_t']
        title = locs_subset.loc[:,'comment'][0] if len(locs_subset['comment'].unique())==1 else 'smth went wrong'
        locs_preview_contours(locs_y, locs_x, locs_subset[color], m, title)


def plot_slow_fast(data, masks_directory):
    for i in data['fov_mask_id'].unique():
        m = show_mask(masks_directory, i, imshow=False)
        locs_subset = data[data['fov_mask_id'] == i]
        title = locs_subset['comment'].unique()[0] if len(locs_subset['comment'].unique()) == 1 else 'smth went wrong'
        
        low_dapp_subset = locs_subset[locs_subset['Dapp'] <= locs_subset['Dapp'].quantile(0.3)]
        high_dapp_subset = locs_subset[locs_subset['Dapp'] >= locs_subset['Dapp'].quantile(0.7)]
        
        locs_preview_contours(low_dapp_subset['y_t'], low_dapp_subset['x_t'], low_dapp_subset['Dapp_log'], m, f'Lowest 30% Dapp - {title}')
        locs_preview_contours(high_dapp_subset['y_t'], high_dapp_subset['x_t'], high_dapp_subset['Dapp_log'], m, f'Highest 30% Dapp - {title}')


def plot_track_length(aggregated_by_track):
    for i in aggregated_by_track['comment'].unique():
        aggregated_by_track.loc[(aggregated_by_track['comment']==i), 'track_length'].hist(bins=50, 
                                                                        cumulative=-1, 
                                                                        density=True, 
                                                                        histtype='step', 
                                                                        label=str(i),
                                                                       )
    plt.ylabel('1-CDF')
    plt.xlabel('track length')
    plt.yscale('log')
    plt.legend()
    sns.move_legend(plt.gca(), 'upper left', bbox_to_anchor=(1, 1), title=None)
    plt.show()


def plot_Dapp_vs_tlength(aggregated_by_track):
    from scipy import stats
    for i in aggregated_by_track['comment'].unique():
    
        tl = aggregated_by_track.loc[aggregated_by_track['comment']==i]['track_length']
        dl = aggregated_by_track.loc[aggregated_by_track['comment']==i]['Dapp_log']
        corr = stats.spearmanr(tl, dl)
        
        values = np.vstack([tl, dl])
        kernel = stats.gaussian_kde(values)(values)
        
        sns.scatterplot(
            data=aggregated_by_track.loc[aggregated_by_track['comment']==i],
            x='track_length', y='Dapp_log',
            c=kernel,
            cmap="viridis",
            # kind='kde', fill=True, 
            alpha=0.7,
        )
        
        sns.kdeplot(
            data=aggregated_by_track.loc[aggregated_by_track['comment']==i],
            x='track_length', y='Dapp_log',
            levels=5,
            alpha=0.2,
        )
    
        plt.xlim(right=tl.quantile(0.98), left=8)
        plt.title(f'{i} \n corr={np.round(corr.correlation, 2)}, p={np.round(corr.pvalue, 4)}')
        plt.show()


def plot_D_overlay(aggregated_by_track):
    ax=sns.kdeplot(aggregated_by_track, x='Dapp_log', hue='comment', fill=True, common_norm=False, # palette='crest',
       alpha=.5, linewidth=0,)
    sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1), title=None)
    plt.show()



def plot_PC_locs(coords_df, 
                 title='', 
                 color='Blues', 
                 size=1, 
                 fig_size=(10, 10), 
                 transparency=0.7,
                ):
    '''
    - takes PC-transformed coordinates and plots scatter (AKA density map)
    - input df has to contain 'pc0' and 'pc1' columns (output from to_PC_raw)
    - title is any string, empty by default
    - color is matplotlib cmap, 'Blues' by default
        
    '''
    
    values = np.vstack([coords_df['pc0'], coords_df['pc1']])
    kernel = stats.gaussian_kde(values)(values)
    
    fig, ax = plt.subplots(figsize=fig_size)
    sns.scatterplot(
    data=coords_df,
    x='pc0', y='pc1',
    c=kernel,
    cmap=color,
    s=size,
    alpha=transparency,
    )
    
    ax.set_aspect('equal', adjustable='box')
    plt.title(title)
    
    plt.show()
    plt.close()


def density_for_feat_range(path_to_file, feature, lower, upper, pixel_um=0.106):
    """
    Plots density maps for defined ranges of selected feature.
    
    - works with single file
    - currently only support plotting for area or Dapp range 
    - assumes pixel size of 0.106 micron, unless specified 

    
    path_to_file : path to file as a string, or from widgets
                   path_to_file=os.path.join(fc.selected_path, chooser_w.value[0])
                   
    feature      : 'area_um_sq' or 'Dapp_log'

    example call:
    from utils.plot import density_for_feat_range
    density_for_feat_range(path_to_file=os.path.join(fc.selected_path, chooser_w.value[0]),                        
                   feature='Dapp_log', 
                   lower=0,
                   upper=0.4,
                  )

    
    """
   
    df = pd.read_csv(path_to_file)
    df['experiment'] = path_to_file[-19:-11] # assumes "tracks_exp_uid42e42_a_uid42e42.csv"-style naming  
    print(path_to_file)
    print(df['experiment'].unique())
    
    # human readable fov id
    df['fov_uids'] = df['experiment']+'_'+df['comment'] 
    
    df.rename(columns={'x_t':'x_t_px', 'y_t':'y_t_px'}, inplace=True)
    
    
    tracks_gb = df.groupby(['experiment', 'mask_uuid', 'cell_id_int', 'particle'])
    cells_gb = df.groupby(['experiment', 'mask_uuid', 'cell_id_int'])
    log_and_print(f'\n{df.shape[0]} locs loaded')
    log_and_print(f'columns: {df.columns} ')
    
    coords = df[['experiment', 
                 'mask_uuid', 
                 'cell_id_int', 
                 'particle', 
                 'fov_mask_id', 
                 'x_t_px', 
                 'y_t_px', 
                 'comment']].copy()
    coords['fov_uids'] = coords['experiment']+'_'+coords['comment']


    # get features
    space = pd.DataFrame()
    
    space['comment']=tracks_gb.agg({'comment':'first'})
    space['experiment']=tracks_gb.agg({'experiment':'first'})
    space['fov_mask_id']=tracks_gb.agg({'fov_mask_id':'first'})
    space['fov_uids']=tracks_gb.agg({'fov_uids':'first'})
    space['Dapp_log']=tracks_gb.agg({'Dapp_log':'first'})
    
    space['area_um_sq']=cells_gb.apply(get_area, pixel=pixel_um).apply(pd.Series)
    
    coords.set_index(['experiment', 'mask_uuid', 'cell_id_int', 'particle'], inplace=True)
    coords[feature]=space[feature]

    if feature=='area_um_sq':
        plt.hist(coords.groupby(coords.index.droplevel('particle')).agg({'area_um_sq':'first'}), 
                 bins=30, color='skyblue', edgecolor='black')
        plt.title('Cell area distribution')
        plt.xlabel('Area (um²)')
        plt.ylabel('n cells')

        plt.axvline(x=lower, color='blue', linestyle='--')
        plt.axvline(x=upper, color='red', linestyle='--')

        plt.show()

    
    elif feature=='Dapp_log':   
        plt.hist(coords.groupby(coords.index).agg({'Dapp_log':'first'}), 
                 bins=30, color='skyblue', edgecolor='black')
        plt.title(r'D$_{app}$ distribution')
        plt.xlabel(r'D$_{app}$')
        plt.ylabel(r'n tracks')

        plt.axvline(x=lower, color='blue', linestyle='--')
        plt.axvline(x=upper, color='red', linestyle='--')

        plt.show()
        
    proceed = input("is that the range you wanted? press 'y' to proceed with plotting")

    if proceed == 'y':
        # scatter
        coords_fov = coords[(coords[feature]>lower)&(coords[feature]<upper)].copy()

        coords_fov.reset_index(inplace=True)
        pcs = coords_fov.groupby(['experiment', 
                               'mask_uuid', 
                               'cell_id_int']).apply(to_PC_raw, 
                                                  pixel=pixel_um, 
                                                  absPC=False)
        print("coords_fov.groupby(['experiment', 'mask_uuid', 'cell_id_int']).apply(to_PC_raw)")
        
        coords_fov[['pc0', 'pc1']] = pcs.reset_index(drop=True)
        title = f'Locs with {feature} in range ({lower},{upper})'
        plot_PC_locs(coords_fov, title=title)

        # marginal distributions
        for dim, label in zip(['pc0', 'pc1'], 
                              ['normalized length', 'normalized width']):
            
            plt.hist(coords_fov[dim], 
                     bins=30, color='skyblue', edgecolor='black')
            plt.xlabel(f'{dim}, {label}')
            plt.ylabel('n localizations') 
            plt.show()
        
        print(f'resulting coords: {coords_fov.shape}, {coords_fov.columns}')
        display(coords_fov.head())
        return coords_fov
        
    else:
        print('rerun the function with different range')
        return 



def plot_2nd_ch_FOV_scatter(fluo, 
                            fig_size=(8, 4), 
                            color_data='value_corr_norm_f',
                            color_map ='viridis',
                           ):
    '''
    fluo:           2nd channel data as pandas df
    fig_size:       figure size in inches
    color_data:     one of ['value_norm', 'd_edge', 'check0', 'check1', 'value_corr_norm_f']
    color_map:      matplotlib cmap
    
    '''
    plt.figure(figsize=fig_size)
    sns.scatterplot(data = fluo, 
                x = 'x_t',
                y = 'y_t',
                hue = color_data, # 'value_norm', 'd_edge', 'check0', 'check1', 'value_corr_norm_f'
                s=3, 
                palette=color_map,
                legend=False,
                edgecolor='none',
                )
    plt.axis('scaled')
    plt.show()
    plt.close()



def plot_2nd_ch_overspill_debug(fluo, 
                            fig_size=(8, 12), 
                            color_map='Greys',
                            marker_size=3,
                           ):
    '''
    fluo:           2nd channel data as pandas df
    fig_size:       figure size in inches
    color_map:      matplotlib cmap
    
    '''
    #'value_corr_norm_f'
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(fig_size))
    
    sns.scatterplot(data=fluo, x='x_t', y='y_t', hue='value_norm', 
                    s=marker_size, palette=color_map, legend=False, edgecolor='none', ax=ax1)
    ax1.axis('scaled')
    ax1.set_title('norm')
    
    sns.scatterplot(data=fluo, x='x_t', y='y_t', hue='value_corr_norm', 
                    s=marker_size, palette=color_map, legend=False, edgecolor='none', ax=ax2)
    ax2.axis('scaled')
    ax2.set_title('corr_norm')

    sns.scatterplot(data=fluo, x='x_t', y='y_t', hue='value_corr_norm_f', 
                    s=marker_size, palette=color_map, legend=False, edgecolor='none', ax=ax3)
    ax3.axis('scaled')
    ax3.set_title('corr_norm_filtered')

    sns.scatterplot(data=fluo, x='x_t', y='y_t', hue='value', 
                    s=marker_size, palette=color_map, legend=False, edgecolor='none', ax=ax4)
    ax4.axis('scaled')
    ax4.set_title('original')
    
    plt.tight_layout()
    plt.show()
    plt.close()




def plot_pc_scatter_with_alpha(df, df_masks, column_name='value_corr_norm_f', marker_size=10, alpha_scale=0.42):
    """
    Plot PC-transformed 2nd channel as scatter on mask scatter.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with fluorescence data (must contain 'pc0_raw', 'pc1_raw', and column_name)
        
    column_name : str, default='value_corr_norm_f'
        Column name to use for alpha transparency and color mapping
        
    marker_size : int, default=10
        Marker size for scatter plots
        
    alpha_scale : float, default=0.42
        Scaling factor for alpha values (multiplied by column values)
    """

    
    fig, ax = plt.subplots()
    
    # plot masks using seaborn with constant alpha
    sns.scatterplot(
        data=df_masks,
        x='pc0_raw', y='pc1_raw',
        color='gray',
        s=marker_size,
        alpha=0.7,
        legend=False,
        edgecolors='none',
        ax=ax
    )
    
    # plot fluos using matplotlib scatter with alpha from column_name
    alphas = np.where(np.isnan(df[column_name]), 0, df[column_name]*alpha_scale)
    # makes pixels where column_name is nan alpha 0 (transparent)
    ax.scatter(
        df['pc0_raw'], df['pc1_raw'],
        c=df[column_name],
        cmap='hot',
        s=marker_size,
        alpha=alphas,
        edgecolors='none'
    )
    
    ax.set_aspect('equal', adjustable='box')
    plt.title('all FOVs together')
    plt.show()
    plt.close()




def plot_pc_scatter_with_kde(
    two_ch_df,
    masks_df,
    column_name='value_corr_norm_f',
    marker_size=20,
    alpha_scatter=0.25,
    cmap='Blues',
    mask_alpha=0.1,
    mask_color='gray',
    kde_levels=7,
    kde_color='blue',
    kde_alpha=0.75,
    kde_linewidth=2.0,
    title=''
    ):
    
    """
    Plot PC-transformed 2nd channel as scatter with masks and weighted KDE isolines.
    
    Parameters
    ----------
    two_ch_df : pandas.DataFrame
        DataFrame with fluorescence data (must contain 'pc0_raw', 'pc1_raw', and column_name)
    masks_df : pandas.DataFrame
        DataFrame with mask data (must contain 'pc0_raw', 'pc1_raw')
    column_name : str, default='value_corr_norm_f'
        Column name to use for coloring and KDE weighting
    marker_size : int, default=20
        Marker size for scatter plots
    alpha_scatter : float, default=0.25
        Alpha transparency for fluorescence scatter plot
    cmap : str, default='Blues'
        Colormap for fluorescence scatter plot
    mask_alpha : float, default=0.1
        Alpha transparency for mask scatter plot
    mask_color : str, default='gray'
        Color for mask scatter plot
    kde_levels : int, default=7
        Number of contour levels for KDE
    kde_color : str, default='blue'
        Color for KDE contours
    kde_alpha : float, default=0.75
        Alpha transparency for KDE contours
    kde_linewidth : float, default=2.0
        Line width for KDE contours
    title : str, default='all FOVs together'
        Plot title
    """

    from scipy.stats import gaussian_kde
    
    fig, ax = plt.subplots()
    
    # plot masks using seaborn with constant alpha
    sns.scatterplot(
        data=masks_df,
        x='pc0_raw', y='pc1_raw',
        color=mask_color,
        s=marker_size,
        alpha=mask_alpha,
        legend=False,
        edgecolors='none',
        ax=ax
    )
    
    # plot fluos using matplotlib scatter
    ax.scatter(
        two_ch_df['pc0_raw'], two_ch_df['pc1_raw'],
        c=two_ch_df[column_name],
        cmap=cmap,
        s=marker_size,
        alpha=alpha_scatter,
        edgecolors='none'
    )
    
    # KDE isolines for column_name
    valid_mask = ~np.isnan(two_ch_df[column_name])
    x_valid = two_ch_df.loc[valid_mask, 'pc0_raw'].values
    y_valid = two_ch_df.loc[valid_mask, 'pc1_raw'].values
    z_valid = two_ch_df.loc[valid_mask, column_name].values
    
    # Prepare grid
    x_min, x_max = x_valid.min(), x_valid.max()
    y_min, y_max = y_valid.min(), y_valid.max()
    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    # Weighted KDE (requires scipy >= 1.10)
    kde = gaussian_kde(np.vstack([x_valid, y_valid]), weights=z_valid)
    f = np.reshape(kde(positions).T, xx.shape)
    
    # Plot weighted KDE contours
    contours = ax.contour(
        xx, yy, f,
        levels=kde_levels,
        colors=kde_color,
        alpha=kde_alpha,
        linewidths=kde_linewidth
    )
    
    ax.set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.show()
    plt.close()



def plot_heatmap_with_mask_contour(
    two_ch_df,
    masks_df,
    column_name='value_corr_norm_f',
    bins=15,
    cmap='inferno',
    gaussian_sigma=0.95,
    otsu_threshold_scale=0.95,
    contour_color='red',
    title=''
    ):
    """
    Plot averaged values as heatmap with mask contour overlay.
    
    Parameters
    ----------
    two_ch_df : pandas.DataFrame
        DataFrame with fluorescence data (must contain 'pc0_raw', 'pc1_raw', and column_name)
    masks_df : pandas.DataFrame
        DataFrame with mask data (must contain 'pc0_raw', 'pc1_raw')
    column_name : str, default='value_corr_norm_f'
        Column name to use for heatmap weighting
    bins : int, default=42
        Number of bins for 2D histogram
    cmap : str, default='inferno'
        Colormap for heatmap
    gaussian_sigma : float, default=0.84
        Sigma for Gaussian filter applied to mask histogram
    otsu_threshold_scale : float, default=0.84
        Scaling factor for Otsu threshold
    contour_color : str, default='red'
        Color for mask contour
    title : str, default='Heatmap with Mask Contour'
        Plot title
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.ndimage import gaussian_filter
    from skimage.filters import threshold_otsu
    
    fig, ax = plt.subplots()
    
    # drop nans
    mask_valid = ~np.isnan(two_ch_df[column_name].values)
    x = two_ch_df['pc0_raw'].values[mask_valid]
    y = two_ch_df['pc1_raw'].values[mask_valid]
    values = two_ch_df[column_name].values[mask_valid]
    
    range_x = x.max() - x.min()
    range_y = y.max() - y.min()
    
    max_range = max(range_x, range_y)
    pad = 0.1 * max_range
    
    center_x = (x.max() + x.min()) / 2
    center_y = (y.max() + y.min()) / 2
    
    xmin, xmax = center_x - (max_range / 2 + pad), center_x + (max_range / 2 + pad)
    ymin, ymax = center_y - (max_range / 2 + pad), center_y + (max_range / 2 + pad)
    
    xbins = np.linspace(xmin, xmax, bins)
    ybins = np.linspace(ymin, ymax, bins)
    
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=[xbins, ybins], weights=values)
    heatmap = heatmap.T  # transpose for imshow
    
    ax.imshow(
        heatmap,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin='lower',
        cmap=cmap,
        aspect='auto'
    )
    
    # Get mask histogram for contour
    mx = masks_df['pc0_raw'].values
    my = masks_df['pc1_raw'].values
    mask_hist, _, _ = np.histogram2d(mx, my, bins=[xbins, ybins])
    
    mask_hist = gaussian_filter(mask_hist, sigma=gaussian_sigma)
    thresh = threshold_otsu(mask_hist[mask_hist > 0]) * otsu_threshold_scale
    ax.contour(
        (xedges[:-1] + xedges[1:]) / 2,
        (yedges[:-1] + yedges[1:]) / 2,
        mask_hist.T,
        levels=[thresh],
        colors=contour_color
    )
    
    ax.set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.show()
    plt.close()



def plot_kde_profiles(
    two_ch_df,
    x_column='pc0_raw',
    weight_column='value_corr_norm_f',
    group_column='cell_id_int',
    kde_kernel=0.63,
    individual_alpha=0.15,
    average_linewidth=3,
    average_color='steelblue',
    figsize=(8, 4),
    xlabel=None,
    ylabel='Density (weighted by value_norm)',
    title='KDE Profiles'
    ):
    """
    Plot KDE profiles with individual cell lines and weighted average.
    
    Parameters
    ----------
    two_ch_df : pandas.DataFrame
        DataFrame with fluorescence data (must contain x_column, weight_column, and group_column)
    x_column : str, default='pc0_raw'
        Column name for x-axis
    weight_column : str, default='value_corr_norm_f'
        Column name for KDE weighting
    group_column : str, default='cell_id_int'
        Column name for grouping individual lines
    kde_kernel : float, default=0.63
        KDE bandwidth adjustment (bw_adjust parameter)
    individual_alpha : float, default=0.15
        Alpha transparency for individual cell KDE lines
    average_linewidth : float, default=3
        Line width for average KDE line
    average_color : str, default='steelblue'
        Color for average KDE line
    figsize : tuple, default=(8, 4)
        Figure size (width, height)
    xlabel : str, optional
        X-axis label. If None, uses x_column name
    ylabel : str, default='Density (weighted by value_norm)'
        Y-axis label
    title : str, default='KDE Profiles'
        Plot title
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=figsize)
    
    # Plot KDE lines for each group with alpha
    sns.kdeplot(
        data=two_ch_df,
        x=x_column,
        weights=weight_column,
        hue=group_column,
        bw_adjust=kde_kernel,
        common_norm=False,
        fill=False,
        alpha=individual_alpha,
        legend=False,
    )
    
    # Plot averaged KDE line for all data combined, thicker and alpha 1
    sns.kdeplot(
        data=two_ch_df,
        x=x_column,
        weights=weight_column,
        bw_adjust=kde_kernel,
        common_norm=False,
        fill=False,
        color=average_color,
        linewidth=average_linewidth,
        alpha=1,
        legend=False,
    )
    
    plt.xlabel(xlabel if xlabel is not None else x_column)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()



def joint_scatter_kde_with_marginals(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    ) -> JointGrid:

    
    """Scatter + 2D KDE + marginal KDEs with legend below the plot."""
    g = sns.JointGrid(
        data=df,
        x=x,
        y=y,
        hue=hue,
        height=7,
        ratio=5,
    )

    # joint: scatter + 2D KDE
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        alpha=0.3,
        ax=g.ax_joint,
    )
    sns.kdeplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        levels=3,
        alpha=0.5,
        ax=g.ax_joint,
        fill=False,
        warn_singular=False,
        legend=False,
    )

    # marginals: 1D KDEs
    sns.kdeplot(
        data=df,
        x=x,
        hue=hue,
        common_norm=False,
        fill=True,
        alpha=0.3,
        ax=g.ax_marg_x,
        legend=False,
    )
    sns.kdeplot(
        data=df,
        y=y,
        hue=hue,
        common_norm=False,
        fill=True,
        alpha=0.3,
        ax=g.ax_marg_y,
        legend=False,
    )

    # legend below, centered, vertical, no title
    sns.move_legend(
        g.ax_joint,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=1,
        title=None,
    )

    plt.tight_layout()
    return g



def plot_embedding(emb_x, emb_y, f_color, figsize=(15, 8), output_folder=''):
    # plots scatter all fovs
    
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
    sns.scatterplot(x=emb_x, y=emb_y, hue=f_color, s=1, palette='hsv', ax=ax1, alpha=0.5, legend=True, edgecolor='none')
    
    # Fix legend marker size
    handles, labels = ax1.get_legend_handles_labels()
    for handle in handles:
        handle.set_markersize(8)  
    ax1.legend(handles, labels, loc='best')
    
    sns.kdeplot(x=emb_x, y=emb_y, fill=True, ax=ax2, cmap='inferno', levels=10)

    plt.tight_layout()

    save_path = os.path.join(output_folder, f'{datetime.today().strftime('%Y-%m-%d')}_emb_{f_color.name}.svg')
    plt.savefig(save_path)
    print(f'image saved in {save_path}')
    
    plt.show()

def plot_embedding_density_alpha(emb_x, emb_y, f_color, figsize=(15, 8), output_folder='', xlim=None, ylim=None, colormap='gist_rainbow', no_axes=False, markersize=4, density_alpha = (0.1, 0.75)):
    """
    Plot UMAP embeddings with density-dependent alpha transparency.
    
    Parameters:
    -----------
    emb_x : array-like
        X-axis coordinates for embedding
    emb_y : array-like
        Y-axis coordinates for embedding
    f_color : Series or array-like
        Color mapping variable (with .name attribute for filename)
    figsize : tuple
        Figure size (width, height)
    output_folder : str
        Directory to save the output image
    xlim : tuple or None
        X-axis limits (xmin, xmax). If None, uses all data
    ylim : tuple or None
        Y-axis limits (ymin, ymax). If None, uses all data
    """
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute point density using kernel density estimation
    from scipy.stats import gaussian_kde
    
    # Create a combined array for density calculation
    xy = np.vstack([emb_x, emb_y])
    kde = gaussian_kde(xy)
    density = kde(xy)**2
    
    # Normalize density to [0.1, 0.75] for alpha values
    density_normalized = (density_alpha[0] + 
                          (density - density.min()) / 
                          (density.max() - density.min()) * 
                          (density_alpha[1] - density_alpha[0]))
    
    # Convert categorical colors to numeric indices
    if pd.api.types.is_categorical_dtype(f_color) or f_color.dtype == 'object':
        color_codes, color_labels = pd.factorize(f_color)
        c_values = color_codes
    else:
        c_values = f_color
    
    # Create scatter plot with density-dependent alpha
    scatter = ax.scatter(emb_x, emb_y, c=c_values, s=markersize, cmap=colormap, 
                        alpha=density_normalized, edgecolor='none')
    
    # Add colorbar with categorical labels
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02, shrink=0.4)
    if pd.api.types.is_categorical_dtype(f_color) or f_color.dtype == 'object':
        # Set colorbar ticks to match categorical values
        cbar.set_ticks(np.arange(len(color_labels)))
        cbar.set_ticklabels(color_labels, fontsize=14)
    cbar.set_label(f_color.name if hasattr(f_color, 'name') else 'Color', rotation=270, labelpad=20)
    
    # Set axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if no_axes:
        ax.axis('off')
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(output_folder, f'{datetime.today().strftime("%Y-%m-%d")}_emb_{f_color.name if hasattr(f_color, "name") else "plot"}.svg')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Image saved to {save_path}')
    
    plt.show()


def plot_emb_single(emb_x, emb_y, all_emb_x, all_emb_y, title, figsize=(5, 5), output_folder=''):
    # plots by FOV, density over general outlines 
    plt.style.use('default')
    fig, ax1 = plt.subplots(figsize=figsize)
        
    sns.kdeplot(x=emb_x, y=emb_y, ax=ax1, fill=True, legend=False, edgecolor='none', color = 'gray') # 'RdYlBu'
    sns.kdeplot(x=all_emb_x, y=all_emb_y, fill=False, ax=ax1, alpha=1, levels = 5, linewidths = 0.75, color='red')
    
    ax1.axis('off')
    plt.title(title)
    plt.tight_layout()

    save_path = os.path.join(output_folder, f'{datetime.today().strftime('%Y-%m-%d')}_semb_{title}.svg')
    plt.savefig(save_path)
    print(f'image saved in {save_path}')
    
    
    plt.show()



def plot_embedding_iter(emb_x, emb_y, f_color, title='', figsize=(15, 8), save=False):
    plt.style.use('dark_background')
    # saves the plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
    sns.scatterplot(x=emb_x, y=emb_y, s=2, palette='Set3', hue=f_color, ax=ax1, legend=False, edgecolor='none', alpha=0.75) # 'RdYlBu'
    sns.kdeplot(x=emb_x, y=emb_y, fill=True, ax=ax2, cmap='inferno', alpha=0.75, levels = 10)
    
    ax1.axis('off')
    ax2.axis('off')
    plt.title(title)
    plt.tight_layout()

    if save:
        plt.savefig('/scratch/project_2009817/SMP_python_images/Taras/temp_output/' + title)
    plt.show()





############## ADA'S HEAT MAPS 
#NEW VERSION - ADA'S GITHUB



def spatial_distribution(filenames, folder = "//home.org.aalto.fi/tomase1/data/Documents/Aalto/Manuscripts/SMT Psychrobacter/DATA/32HS/22DEG", feature = "Dapp_log", pixel_um=0.106):
    coords_ls = []
    locs_sum = []
    print(filenames)
    for i, p in enumerate(filenames):
        print(p)
        print(f"{folder}/{p}")
        df = pd.read_csv(f"{folder}/{p}")
        df['experiment'] = f"experiment{i}"
        print(df['experiment'].unique())
        
        # human readable fov id
        df['fov_uids'] = df['experiment']+'_'+df['comment'] 
        
        df.rename(columns={'x_t':'x_t_px', 'y_t':'y_t_px'}, inplace=True)
        
        
        tracks_gb = df.groupby(['experiment', 'mask_uuid', 'cell_id_int', 'particle'])
        cells_gb = df.groupby(['experiment', 'mask_uuid', 'cell_id_int'])
        log_and_print(f'\n{df.shape[0]} locs loaded')
        locs_sum.append(df.shape[0])
        log_and_print(f'columns: {df.columns} ')
        
        coords = df[['experiment', 'mask_uuid', 'cell_id_int', 'particle', 'fov_mask_id', 'x_t_px', 'y_t_px', 'comment']].copy()
        coords['fov_uids'] = coords['experiment']+'_'+coords['comment']
    
    
        # get feature
        space = pd.DataFrame()
        
        space['comment']=tracks_gb.agg({'comment':'first'})
        space['experiment']=tracks_gb.agg({'experiment':'first'})
        space['fov_mask_id']=tracks_gb.agg({'fov_mask_id':'first'})
        space['fov_uids']=tracks_gb.agg({'fov_uids':'first'})
        space['Dapp_log']=tracks_gb.agg({'Dapp_log':'first'})
        
        space['area_um_sq']=cells_gb.apply(get_area, pixel=pixel_um).apply(pd.Series)
        
        coords.set_index(['experiment', 'mask_uuid', 'cell_id_int', 'particle'], inplace=True)
        coords[feature]=space[feature]
        coords["area"] = space["area_um_sq"]
        coords = coords[(coords["area"]<2.5)]
        coords_ls.append(coords)
    print(locs_sum)


    up_lim = []
    low_lim = []
    for coor in coords_ls:
        low = float(input("Type in the lower limit for Dapp:"))

        high = float(input("Type in the upper limit Dapp:"))
        
        plt.hist(coor.groupby(coor.index).agg({'Dapp_log':'first'}), 
                 bins=30, color='skyblue', edgecolor='black')
        plt.title(r'D$_{app}$ distribution')
        plt.xlabel(r'D$_{app}$')
        plt.ylabel(r'n tracks')

        plt.axvline(x=low, color='blue', linestyle='--')
        plt.axvline(x=high, color='red', linestyle='--')

        low_lim.append(low)
        up_lim.append(high)
        print(up_lim)

        plt.show()
        
    proceed = input("Are these the ranges you wanted? Press 'y' to proceed with plotting")
    print("Lower limits:",low_lim)
    print("Upper Limits:",up_lim)
    pc_mask_ls = []
        
    if proceed == 'y':   
        coords_fovs = []
        
        for idx, coords in enumerate(coords_ls):
            lower = low_lim[idx]
            upper = up_lim[idx]
            coords_limit = coords[(coords[feature]>lower)&(coords[feature]<upper)].copy()
            pc_mask = get_pc_transformed_mask_coords(coords_limit, flip_off=True)
            pc_mask_ls.append(pc_mask)
            
            coords_limit.reset_index(inplace=True)
            pcs = coords_limit.groupby(['experiment', 
                                   'mask_uuid', 
                                   'cell_id_int', 
                                   'particle']).apply(to_PC_raw, 
                                                      pixel=pixel_um, 
                                                      absPC=False)
            
            coords_limit[['pc0', 'pc1']] = pcs.reset_index(drop=True)
            coords_fovs.append(coords_limit)

        print(len(pc_mask_ls))

        counter = 0
        for filename, mask in zip(filenames, pc_mask_ls):
            mask.to_csv(f"{folder}/{filename}_{counter}_pc_mask_data.csv")
            counter += 1
        
        get_heatmap_and_contour_mask(coords_fovs, pc_mask_ls, folder, filenames, locs_sum)
   
    else:
    
        print('rerun the function with different range')
        return 
    

        

def get_heatmap_and_contour_mask(coords_fovs, pc_mask_ls, folder, filenames, locs_sum, bins=15, feature = "Dapp_log"):

    masked_heatmaps = []
    mask_contours = []
    threshold = []
    mask_hists = []
    comps = []
    
    #######
    """
    Calculate the heatmap for the first FOV in the filenames list, all other FOVs/heatmaps will be compared to this one when calculating
    the difference heatmaps
    """
    comp = coords_fovs[0]
   
    comp['pc0']= abs(comp['pc0'])
    comp['pc0']= (comp['pc0']) * -1

    comps.append(comp)

    comp1 = coords_fovs[1]
    comp1['pc0']= abs(comp1['pc0'])
    comp1['pc0']= (comp1['pc0']) * -1
    comps.append(comp1)

    counter = 0
    for filename, df in zip(filenames, comps):
        df.to_csv(f"{folder}/{filename}_{counter}_pc_data.csv")
        counter += 1
  
    
    x_ref = comp['pc0']
    y_ref = comp['pc1']
  
    # choose a symmetric range around 0 that covers your data (add a bit of pad)
    Y = max(abs(y_ref.min()), abs(y_ref.max()))
    Y *= 1.1  # optional pad
    
    yedges_ref = np.linspace(-Y, Y, bins + 1)  # centers will include 0
    
    # match x bin width to y 
    dx_ref = yedges_ref[1] - yedges_ref[0]

    xmin_ref = float(x_ref.min())
    # left edge aligned to dx, right edge at 0
    x_left = np.floor(xmin_ref / dx_ref) * dx_ref
    xedges_ref = np.arange(x_left, 0.0 + dx_ref, dx_ref)  # includes the 0.0 right edge
    
    heatmap_ref, xedges_ref, yedges_ref = np.histogram2d(x_ref, y_ref, bins=[xedges_ref, yedges_ref])
    heatmap_ref = heatmap_ref.T

        # Get mask histogram for contour
    mask_ref = pc_mask_ls[0]
    mask_ref['pc0_raw'] = abs(mask_ref['pc0_raw'])
    mask_ref['pc0_raw'] = (mask_ref['pc0_raw'])*-1
    
    mx_ref = mask_ref['pc0_raw'].values
    my_ref = mask_ref['pc1_raw'].values
    
    mask_hist_ref, _, _ = np.histogram2d(mx_ref, my_ref, bins=[xedges_ref, yedges_ref])
    
    mask_hist_ref = gaussian_filter(mask_hist_ref, sigma=0.5, mode ="reflect")
    mask_hist_ref[-1, :] = mask_hist_ref[-2, :]

    thresh_ref = threshold_otsu(mask_hist_ref[mask_hist_ref > 0]) * 0.5
    
       # After computing thresh_ref
    inside_grid_ref = mask_hist_ref >= thresh_ref      # (Ny, Nx) on centers
    inside_img_ref = inside_grid_ref.T                 # (Ny, Nx) matches imshow

    heatmap_masked_ref = np.where(inside_img_ref, heatmap_ref, 0.0)
    
    
    inside_img_ref = inside_grid_ref.T
    
    heatmap_masked_ref = np.where(inside_img_ref, heatmap_ref, 0.0)
    
    masked_heatmaps.append(heatmap_masked_ref)
    

    # Plot masked heatmap

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        heatmap_masked_ref,
        extent=[xedges_ref[0], xedges_ref[-1], yedges_ref[0], yedges_ref[-1]],
        origin='lower',
        cmap='viridis',
        aspect='equal'
    )

    contour = ax.contour( (xedges_ref[:-1] + xedges_ref[1:]) / 2, (yedges_ref[:-1] + yedges_ref[1:]) / 2, mask_hist_ref.T, levels=[thresh_ref], colors="red" )
     # Add a color bar with labeling
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Number of Localisations')
    ax.set_aspect('equal', adjustable='box')
    plt.title("Heatmap testing")

    plt.ylim(-0.8, 0.8) #heatmap size
    plt.xlim(left = -1.1, right= 0.0)
    plt.savefig(f"{folder}/heatmap_reference.svg", dpi = 300)
    plt.show()
    plt.close()
    
    #plt.imshow(inside_img_ref)
    #plt.show()
    #plt.close()
    mask_contours.append(inside_img_ref)
  
    bins_rel = 30
    
    for i, df in enumerate(comps):
        for dim, label in zip(['pc0', 'pc1'], ['normalized length', 'normalized width']):
            plt.figure()
    
            sns.histplot(
                data=df, x=dim,
                bins=bins_rel, color='slateblue',
                stat='density', alpha=0.25, edgecolor=None
            )
            sns.kdeplot(
                data=df, x=dim,
                color='black', lw=1.8, bw_adjust=1.0
            )
    
            if dim == "pc1":
                plt.xlim(-0.8, 0.8)
            
            elif dim == "pc0":
                plt.ylim(0, 2.0)
                plt.xlim(left = -1.2)
    
            plt.xlabel(f'{dim}, {label}')
            plt.ylabel('Probability density')
            plt.title(f'comp #{i} — {label}')
            plt.savefig(f"{folder}/1D_plot{label}_comp{i}.svg", dpi = 300)
            plt.show()  # or move outside to show all at once
    
    
    for idx, coords_fov in enumerate(coords_fovs[1:]):

        x = coords_fov['pc0']
        y = coords_fov['pc1']


        heatmap, _, _ = np.histogram2d(x, y, bins=[xedges_ref, yedges_ref])
        heatmap = heatmap.T

        plt.imshow(heatmap)
        plt.show()
        print("Heatmap raw")

        # Get mask histogram for contour
        pc_mask_ls[idx+1]['pc0_raw'] = abs(pc_mask_ls[idx+1]['pc0_raw'])
        pc_mask_ls[idx+1]['pc0_raw'] = (pc_mask_ls[idx+1]['pc0_raw'])*-1
        
        mx = pc_mask_ls[idx+1]['pc0_raw'].values
        my = pc_mask_ls[idx+1]['pc1_raw'].values
       
        mask_hist, _, _ = np.histogram2d(mx, my, bins=[xedges_ref, yedges_ref])
        
        mask_hist = gaussian_filter(mask_hist, sigma=0.5, mode = "reflect")
        mask_hist[-1, :] = mask_hist[-2, :]
        #plt.imshow(mask_hist)
        #plt.show()
    
        thresh = threshold_otsu(mask_hist[mask_hist > 0]) * 0.5
        
    
        # Build mask on the grid (same orientation as heatmap before transpose)
        inside_grid = mask_hist >= thresh            # shape: (Nx, Ny)
    
        # heatmap for imshow is transposed (Ny, Nx), so transpose the mask too
        inside_img = inside_grid.T                   # shape matches heatmap.T
        
        mask_contours.append(inside_img)

        plt.imshow(inside_img)
        plt.show()
        
        heatmap_masked = np.where(inside_img, heatmap, 0.0)
        masked_heatmaps.append(heatmap_masked)


        # Plot masked heatmap
        fig, aax = plt.subplots(figsize=(8,6))
        im = aax.imshow(
            heatmap_masked,
            extent=[xedges_ref[0], xedges_ref[-1], yedges_ref[0], yedges_ref[-1]],
            origin='lower',
            cmap='viridis',
            aspect='equal'
        )
        aax.contour(
            (xedges_ref[:-1] + xedges_ref[1:]) / 2,
            (yedges_ref[:-1] + yedges_ref[1:]) / 2,
            mask_hist.T,
            levels=[thresh],
            colors="red"
        )
         # Add a color bar with labeling
        cbar = fig.colorbar(im, ax=aax)
        cbar.set_label('Number of Localisations')
        plt.title(f"Heatmap {idx+1}")
        plt.ylim(-0.8, 0.8) #heatmap size
        plt.xlim(left =-1.1, right= 0.0)
        plt.savefig(f"{folder}/heatmap_{idx+1}.svg", dpi = 300)
        plt.show()
    
    #Plot difference heatmaps

    heatmap1_norm = masked_heatmaps[0]/ masked_heatmaps[0].sum()


    for idx, heatmap in enumerate(masked_heatmaps[1:]):
        heatmap2_norm = heatmap/ heatmap.sum()

        # Compute the difference of masked heatmaps
        difference_heatmap = heatmap1_norm - heatmap2_norm



    
        v = np.nanmax(np.abs(difference_heatmap))



        heatmap_masked = np.where(inside_img_ref, difference_heatmap, 0.0)


    
        # Visualize masked heatmap
        fig, ax = plt.subplots(figsize=(8,6))
        
        cax = ax.imshow(
            heatmap_masked,
            extent=[xedges_ref[0], xedges_ref[-1], yedges_ref[0], yedges_ref[-1]],
            origin='lower',
            aspect='auto',
            cmap='bwr',
            vmin=-v,
            vmax=v)
    
    
        contour = ax.contour(
        (xedges_ref[:-1] + xedges_ref[1:]) / 2,
        (yedges_ref[:-1] + yedges_ref[1:]) / 2,
        mask_hist_ref.T,
        levels=[thresh_ref],
        colors="red"
        )
        
        
        # Add a color bar with labeling
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label('Localisation Probability')
        plt.ylim(-0.8, 0.8) # heatmap size
        plt.xlim(left =-1.1, right= 0.0)
        ax.set_aspect('equal', adjustable='box')
        plt.title("Difference Heatmap within contour")
        plt.savefig(f"{folder}/heatmap_difference{idx+1}.svg", dpi = 300)
        plt.show()


        
    #Plot relative densities
    dens = []
    
    
    for idx, heatmap in enumerate(masked_heatmaps):
        print(locs_sum[idx])

        #Divide heatmap by total sum of locs in distribution
        heatmap_sum = heatmap/locs_sum[idx]   

        print(type(heatmap_sum))
        
        heatmap_arr = heatmap_sum.ravel()
        heatmap_arr = np.delete(heatmap_arr, np.where(heatmap_arr == 0))

        dens.append(heatmap_arr)


    #Plotting
    # Labels: S0, S1, ...
    states = [f"S{i}" for i in range(len(masked_heatmaps))]
    temperature = [f"HS" for i in range(len(masked_heatmaps))]
    
    # Build long-form DataFrame for seaborn
    parts = [pd.DataFrame({"value": arr, "state": s}) for s, arr in zip(states, dens)]
    df = pd.concat(parts, ignore_index=True)

    
    # Plot ECDFs with labels
    plt.figure(figsize=(8, 6))
    sns.ecdfplot(data=df, x="value", hue="state")  # one curve per state
    plt.xlabel("Relative density")
    plt.ylabel("ECDF")
    plt.ylim(0, 1.0)
    #plt.legend(title="State")

    plt.show()

    




def get_pc_transformed_mask_coords(fluo, flip_off=True):
    '''
    fluo:       Dataframe containing paths and relevant data for multiple masks
                and a column 'fov_mask_id' to identify each mask uniquely.
    flip_off:   Boolean to randomly flip the signs of transformed coordinates
    Returns:    Dataframe with PC-transformed mask coordinates for all masks
    '''

    # Initialize a list to store transformed coordinates for all masks
    all_masks = []

    # Iterate over each unique mask identifier in the 'fov_mask_id' column
    for fov_mask_id in fluo['fov_mask_id'].unique():
        # Filter the dataframe for the current fov_mask_id
        
        # Get mask
        directory = os.getenv('SMPP_DATA_DIR')+'/analysis/masks_omni/'

        m = show_mask(directory, fov_mask_id, imshow=False)

        # Transform and aggregate masks for the current mask file
        masks = []
        for i in range(1, np.max(m) + 1):
            cm = (m == i)  # cell mask

            # Create mask coordinates dataframe
            mask_coords = pd.DataFrame()
            mask_coords['x_t'] = np.where(cm)[0]
            mask_coords['y_t'] = np.where(cm)[1]
            
            # Assign cell ID
            mask_coords['cell_id_int'] = i
            
            # PC transform mask coordinates
            from sklearn.decomposition import PCA
            
            pca = PCA(n_components=2)
            pca_cm_coords = pca.fit_transform(mask_coords[['x_t', 'y_t']])

            # Optionally scale and flip the transformed coordinates
            scaling = np.max(np.abs(pca_cm_coords))
            mask_coords['pc0_raw'] = pca_cm_coords[:, 0] / scaling
            mask_coords['pc1_raw'] = pca_cm_coords[:, 1] / scaling
            
            if flip_off:
                mask_coords['pc0_raw'] = mask_coords['pc0_raw'] * np.random.choice([1, -1])
                mask_coords['pc1_raw'] = mask_coords['pc1_raw'] * np.random.choice([1, -1])
            
            # Add mask file identifier to the dataframe
            mask_coords['fov_mask_id'] = fov_mask_id
            
            # Append individual mask coordinates to the list
            masks.append(mask_coords)
        
        # Concatenate masks for the current mask file and append to all_masks
        all_masks.extend(masks)
    
    # Concatenate all mask dataframes into one final dataframe
    result_df = pd.concat(all_masks, ignore_index=True)
    
    return result_df

        

def ready_pc_heatmaps(folder, coord_paths, mask_paths, gaussian = 0.8, otsu = 0.5, bins = 15):
    pc_coords = []
    pc_masks = []
    masked_heatmaps = []
    mask_contours = []
    threshold = []
    mask_hists = []
    pc_coords_filt = []

    print(coord_paths)
    for coord, mask in zip(coord_paths, mask_paths):
        print(f"{folder}/{coord}")
        df_coords = pd.read_csv(f"{folder}/{coord}")
        pc_coords.append(df_coords)
        df_masks = pd.read_csv(f"{folder}/{mask}")
        pc_masks.append(df_masks)

    print(len(pc_coords))
    up_lim = []
    low_lim = []
    
    for coord in pc_coords:
        low = float(input("Type in the lower limit for Dapp:"))
    
        high = float(input("Type in the upper limit Dapp:"))
        
        plt.hist(coord.groupby(coord.index).agg({'Dapp_log':'first'}), 
                 bins=30, color='skyblue', edgecolor='black')
        plt.title(r'D$_{app}$ distribution')
        plt.xlabel(r'D$_{app}$')
        plt.ylabel(r'n tracks')
    
        plt.axvline(x=low, color='blue', linestyle='--')
        plt.axvline(x=high, color='red', linestyle='--')
    
        low_lim.append(low)
        up_lim.append(high)
        print(up_lim)
    
        plt.show()

        
    proceed = input("Are these the ranges you wanted? Press 'y' to proceed with plotting")
    if proceed == 'y': 
        print("Lower limits:",low_lim)
        print("Upper Limits:",up_lim)
    
        for idx, coords in enumerate(pc_coords):
            lower = low_lim[idx]
            print("This is lower", lower)
            upper = up_lim[idx]
            filtered = coords[(coords["Dapp_log"]>lower)&(coords["Dapp_log"]<upper)].copy()
            #print(filtered.head(20))
            pc_coords_filt.append(filtered)
        

    else:
        print("Run again with correct limits")


    
    #######

    #Calculate the heatmap for the first FOV in the filenames list, all other FOVs/heatmaps will be compared to this one when calculating
    #the difference heatmaps

    print(len(pc_coords_filt[0]))
    print(len(pc_coords_filt[1]))
    
    reference = pc_coords_filt[0]
    
    x_ref = reference['pc0']
    y_ref = reference['pc1']

    # choose a symmetric range around 0 that covers your data (add a bit of pad)
    Y = max(abs(y_ref.min()), abs(y_ref.max()))
    Y *= 1.1  # optional pad
    
    yedges_ref = np.linspace(-Y, Y, bins + 1)  # centers will include 0
    
    # match x bin width to y 
    dx_ref = yedges_ref[1] - yedges_ref[0]

    xmin_ref = float(x_ref.min())
    # left edge aligned to dx, right edge at 0
    x_left = np.floor(xmin_ref / dx_ref) * dx_ref
    xedges_ref = np.arange(x_left, 0.0 + dx_ref, dx_ref)  # includes the 0.0 right edge
    
    heatmap_ref, xedges_ref, yedges_ref = np.histogram2d(x_ref, y_ref, bins=[xedges_ref, yedges_ref])
    heatmap_ref = heatmap_ref.T

    # Get mask histogram for contour
    mask_ref = pc_masks[0]
    mask_ref['pc0_raw'] = abs(mask_ref['pc0_raw'])
    mask_ref['pc0_raw'] = (mask_ref['pc0_raw'])*-1
    
    mx_ref = mask_ref['pc0_raw'].values
    my_ref = mask_ref['pc1_raw'].values
    
    mask_hist_ref, _, _ = np.histogram2d(mx_ref, my_ref, bins=[xedges_ref, yedges_ref])
    
    mask_hist_ref = gaussian_filter(mask_hist_ref, sigma=0.5, mode ="reflect")
    mask_hist_ref[-1, :] = mask_hist_ref[-2, :]

    thresh_ref = threshold_otsu(mask_hist_ref[mask_hist_ref > 0]) * 0.5
    
       # After computing thresh_ref
    inside_grid_ref = mask_hist_ref >= thresh_ref      # (Ny, Nx) on centers
    inside_img_ref = inside_grid_ref.T                 # (Ny, Nx) matches imshow

    heatmap_masked_ref = np.where(inside_img_ref, heatmap_ref, 0.0)
    
    
    inside_img_ref = inside_grid_ref.T
    
    heatmap_masked_ref = np.where(inside_img_ref, heatmap_ref, 0.0)
    
    masked_heatmaps.append(heatmap_masked_ref)
    

    # Plot masked heatmap

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        heatmap_masked_ref,
        extent=[xedges_ref[0], xedges_ref[-1], yedges_ref[0], yedges_ref[-1]],
        origin='lower',
        cmap='viridis',
        aspect='equal'
    )

    contour = ax.contour( (xedges_ref[:-1] + xedges_ref[1:]) / 2, (yedges_ref[:-1] + yedges_ref[1:]) / 2, mask_hist_ref.T, levels=[thresh_ref], colors="red" )
     # Add a color bar with labeling
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Number of Localisations')
    ax.set_aspect('equal', adjustable='box')
    plt.title("Heatmap testing")

    plt.ylim(-0.4, 0.4)
    plt.xlim(left = -1.1, right= 0.0)
    plt.savefig(f"{folder}/heatmap_reference.svg", dpi = 300)
    plt.show()
    plt.close()
    
    #plt.imshow(inside_img_ref)
    #plt.show()
    #plt.close()
    mask_contours.append(inside_img_ref)
  
    bins_rel = 30
    
    for i, df in enumerate(pc_coords_filt):
        for dim, label in zip(['pc0', 'pc1'], ['normalized length', 'normalized width']):
            plt.figure()
    
            sns.histplot(
                data=df, x=dim,
                bins=bins_rel, color='slateblue',
                stat='density', alpha=0.25, edgecolor=None
            )
            sns.kdeplot(
                data=df, x=dim,
                color='black', lw=1.8, bw_adjust=1.0
            )
    
            if dim == "pc1":
                plt.xlim(-0.8, 0.8)
            
            elif dim == "pc0":
                plt.ylim(0, 2.0)
                plt.xlim(left = -1.2)
    
            plt.xlabel(f'{dim}, {label}')
            plt.ylabel('Probability density')
            plt.title(f'comp #{i} — {label}')
            #plt.savefig(f"{folder}/1D_plot{label}_comp{i}.svg", dpi = 300)
            plt.show()  # or move outside to show all at once
    
    
    for idx, coords_fov in enumerate(pc_coords_filt[1:]):

        x = coords_fov['pc0']
        y = coords_fov['pc1']


        heatmap, _, _ = np.histogram2d(x, y, bins=[xedges_ref, yedges_ref])
        heatmap = heatmap.T

        plt.imshow(heatmap)
        plt.show()
        print("Heatmap raw")

        # Get mask histogram for contour
        pc_masks[idx+1]['pc0_raw'] = abs(pc_masks[idx+1]['pc0_raw'])
        pc_masks[idx+1]['pc0_raw'] = (pc_masks[idx+1]['pc0_raw'])*-1
        
        mx = pc_masks[idx+1]['pc0_raw'].values
        my = pc_masks[idx+1]['pc1_raw'].values
       
        mask_hist, _, _ = np.histogram2d(mx, my, bins=[xedges_ref, yedges_ref])
        
        mask_hist = gaussian_filter(mask_hist, sigma=0.5, mode = "reflect")
        mask_hist[-1, :] = mask_hist[-2, :]


    
        thresh = threshold_otsu(mask_hist[mask_hist > 0]) * 0.5
        
    
        # Build mask on the grid (same orientation as heatmap before transpose)
        inside_grid = mask_hist >= thresh            # shape: (Nx, Ny)
    
        # heatmap for imshow is transposed (Ny, Nx), so transpose the mask too
        inside_img = inside_grid.T                   # shape matches heatmap.T
        
        mask_contours.append(inside_img)

        plt.imshow(inside_img)
        plt.show()
        
        heatmap_masked = np.where(inside_img, heatmap, 0.0)
        masked_heatmaps.append(heatmap_masked)

        print(np.array_equal(masked_heatmaps[0], masked_heatmaps[1]))


        # Plot masked heatmap
        fig, aax = plt.subplots(figsize=(8,6))
        im = aax.imshow(
            heatmap_masked,
            extent=[xedges_ref[0], xedges_ref[-1], yedges_ref[0], yedges_ref[-1]],
            origin='lower',
            cmap='viridis',
            aspect='equal'
        )
        aax.contour(
            (xedges_ref[:-1] + xedges_ref[1:]) / 2,
            (yedges_ref[:-1] + yedges_ref[1:]) / 2,
            mask_hist.T,
            levels=[thresh],
            colors="red"
        )
                # Add a color bar with labeling
        cbar = fig.colorbar(im, ax=aax)
        cbar.set_label('Number of Localizations')
        plt.title(f"Heatmap {idx+1}")
        plt.ylim(-0.4, 0.4)
        plt.xlim(left =-1.1, right= 0.0)
        plt.savefig(f"{folder}/heatmap_{idx+1}.svg", dpi = 300)
        plt.show()
    
    #Plot difference heatmaps

    heatmap1_norm = masked_heatmaps[0]/ masked_heatmaps[0].sum()
    #print(heatmap1_norm)


    for idx, heatmap in enumerate(masked_heatmaps[1:]):
        heatmap2_norm = heatmap/ heatmap.sum()

        # Compute the difference of masked heatmaps
        difference_heatmap = heatmap1_norm - heatmap2_norm



    
        v = np.nanmax(np.abs(difference_heatmap))

        #Create contour from average of binary heatmap masks
        
        heatmap_masked = np.where(inside_img_ref, difference_heatmap, 0.0)
          
    
        # Visualize masked heatmap
        fig, ax = plt.subplots(figsize=(8,6))
        
        cax = ax.imshow(
            heatmap_masked,
            extent=[xedges_ref[0], xedges_ref[-1], yedges_ref[0], yedges_ref[-1]],
            origin='lower',
            aspect='auto',
            cmap='bwr',
            vmin=-v,
            vmax=v)
    
    
        contour = ax.contour(
        (xedges_ref[:-1] + xedges_ref[1:]) / 2,
        (yedges_ref[:-1] + yedges_ref[1:]) / 2,
        mask_hist_ref.T,
        levels=[thresh_ref],
        colors="red"
        )
        
        
        # Add a color bar with labeling
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label('Localisation Probability')
        plt.ylim(-0.4, 0.4)
        plt.xlim(left =-1.1, right= 0.0)
        ax.set_aspect('equal', adjustable='box')
        plt.title("Difference Heatmap within contour")
        plt.savefig(f"{folder}/heatmap_difference{idx+1}.svg", dpi = 300)
        plt.show()

    
   





        
    