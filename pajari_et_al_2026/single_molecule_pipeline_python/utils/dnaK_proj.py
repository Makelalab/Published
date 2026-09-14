import tifffile
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib
from PIL import Image
import boto3
import os
import numpy as np
import pandas as pd
import trackpy as tp
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.segmentation import find_boundaries
from utils.tiff_writer import download_img_allas, show_mask

from scipy import stats
from matplotlib.colors import ListedColormap
from scipy.ndimage import label, find_objects
from skimage import measure

from utils.xml_writer import get_exps_meta_from_xml
from utils.logging import setup_logger, log_and_print, save_as_pickle
from utils.get_features import get_area, to_PC_raw

from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from scipy.stats import gaussian_kde




def bandpass(image, noise_size=1, smoothing_size=7):
    img = image.astype(np.float32)
    low = gaussian_filter(img, sigma=noise_size, mode='reflect')
    high = uniform_filter(img, size=smoothing_size, mode='reflect')
    return np.clip(low - high, a_min=0, a_max=None)

def detect_aggregates_in_cells(
    image,
    cell_mask,
    diameter =(9,9),
    minmass=None,
    noise_size=1,
    smoothing_size=7,
    threshold=130,
    characterize=True,
    engine='python',
    preprocess=False):
    
    
    assert image.shape == cell_mask.shape, f"Shape mismatch: image {image.shape}, mask {cell_mask.shape}"
    # Optional prefiltering (helps SNR; you can also pass image directly to tp.locate)
    filt = bandpass(image, noise_size=noise_size, smoothing_size=smoothing_size)
    plt.imshow(filt)
    plt.show()
    # If minmass not given, pick a heuristic based on percentile
    if minmass is None:
        # Trackpy's 'mass' relates to summed brightness in the spot; tune the percentile
        # 99th is often okay; adjust to your data
        minmass = np.percentile(image, 99)

    # Detect spots in a single 2D frame
    feats = tp.locate(
        filt, diameter=diameter,
        minmass=float(minmass),
        noise_size=noise_size,
        smoothing_size=smoothing_size,
        threshold=int(threshold),
        characterize=characterize,
        preprocess = preprocess,
        engine=engine
    )

    
    
    if feats is None or len(feats) == 0:
        # No spots detected
        feats = pd.DataFrame(columns=['x', 'y', 'mass', 'size', 'ecc', 'signal', 'raw_mass'])
        feats['cell_id_int'] = pd.Series(dtype=int)

    # Map subpixel coords (x, y) to cell IDs from the mask
    # Trackpy uses (x, y) in pixel units; convert to integer pixel indices
    # Clip to image bounds to avoid out-of-range
    if len(feats) > 0:
        yy = np.clip(np.rint(feats['y']).astype(int), 0, cell_mask.shape[0]-1)
        xx = np.clip(np.rint(feats['x']).astype(int), 0, cell_mask.shape[1]-1)
        #print(xx, yy)
        cell_ids = cell_mask[yy, xx]
        feats = feats.assign(cell_id_int=cell_ids)
        # Keep only spots that fall inside labeled cells (cell_id > 0)
        feats = feats[feats['cell_id_int'] > 0].reset_index(drop=True)

    # Per-cell counts
    labeled_cells = np.unique(cell_mask)
    labeled_cells = labeled_cells[labeled_cells > 0]  # exclude background
    counts = feats.groupby('cell_id_int').size().reindex(labeled_cells, fill_value=0)
    cells_with = (counts > 0).sum()
    cells_without = (counts == 0).sum()

    summary = {
        'n_cells_total': int(len(labeled_cells)),
        'n_cells_with_aggregates': int(cells_with),
        'n_cells_without_aggregates': int(cells_without)
    }

    return feats, counts, summary





def get_dist_to_agg(
    df: pd.DataFrame,
    coords: pd.DataFrame,
    mask: pd.DataFrame,
    show_plots: bool = True,          # turn plotting on/off
    color_by_nearest_ref: bool = False,  # color locs by nearest reference index
    pixel = 0.106 #pixel size in micrometers
) -> pd.DataFrame:   
    
    '''
    This function computes, for each localization in df, the Euclidean distance to the nearest reference point from coords within the same cell region defined by a labeled mask,
    optionally plotting per-cell overlays. Required columns: df must have 'cell_id_int', 'x', 'y' (and optionally 'Dapp_log', 'comment'); coords must have 'cell_id_int', 'x', 'y';
    mask is a 2D labeled image where mask == cell_id isolates a cell. For each cell_id in coords, it filters the mask, skips cells with area > 2.5 µm², gathers that cell’s localizations from df,
    computes pairwise distances to its reference points, and records the nearest distance and reference index. The output is a per-localization DataFrame with 'x', 'y', 'cell_id_int', 
    'nearest_ref_dist' (pixels). Optional arguments: show_plots to display per-cell overlays and color_by_nearest_ref to color localizations by nearest reference index; distances are scaled using pixel (µm/pixel)
    '''

    df = df.copy()
    df[['x', 'y']] = df[['y', 'x']].to_numpy()

    coords = coords.copy()
    
    print("1")
    out = []
    
    for cell_id in coords['cell_id_int'].unique():
        m = (mask==cell_id)
        area_um_sq = area_um_sq = np.sum(m)*pixel**2

        
        if area_um_sq >2.5:
            continue
        # reference points for this cell_id (can be multiple)
        refs = coords.loc[coords['cell_id_int'] == cell_id, ['x', 'y']].to_numpy(dtype=float)  # shape (r, 2)

        if refs.size == 0:
            continue

            
    
        # target points for this cell_id

        locs_df = df.loc[df['cell_id_int'] == cell_id, ['x', 'y', "Dapp_log", "comment"]]
        if locs_df.empty:
            continue
  
        pts = locs_df[['x', 'y']].to_numpy(dtype=float) # shape (n, 2)

    
        # pairwise distances: (n, r)
        D = np.linalg.norm(pts[:, None, :] - refs[None, :, :], axis=2)

        # example: nearest reference per point
        nearest_dist = D.min(axis=1)
        nearest_ref_idx = D.argmin(axis=1)

        if show_plots:
            fig, ax = plt.subplots(figsize=(5, 5))
    
            # Show only the current cell mask
            mask_bin = (mask == cell_id)
    

            im = ax.imshow(mask_bin, cmap='Greys', alpha=0.3, origin='lower')  

    
            if color_by_nearest_ref:
                sc = ax.scatter(pts[:, 0], pts[:, 1], c=nearest_ref_idx, s=12, cmap='tab10', alpha=0.8, label='locs')
                cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('nearest_ref_idx')
            else:
                ax.scatter(pts[:, 0], pts[:, 1], s=5, color='C0', alpha=0.8, label='locs')
    
            ax.scatter(refs[:, 0], refs[:, 1], s=50, marker='x', color='C3', linewidths=2, label='refs')
    
            ax.set_title(f'cell_id {cell_id}')
            ax.set_aspect('equal', adjustable='box')
            ax.legend(loc='best')
            plt.tight_layout()
            plt.show()
    
        # store/attach back to the original rows
        tmp = locs_df.copy()
        tmp['nearest_ref_dist'] = nearest_dist
        tmp['nearest_ref_idx'] = nearest_ref_idx
        tmp['nearest_ref_dist_um'] = tmp['nearest_ref_dist'] * pixel
        tmp['nearest_ref_dist_nm'] = tmp['nearest_ref_dist_um'] *1000
        tmp['cell_id_int'] = cell_id
        out.append(tmp)

            # Combine all cells
    res = pd.concat(out, ignore_index=True)
    
    # drop rows with missing values that would break plotting
    res = res.dropna(subset=['Dapp_log', 'nearest_ref_dist'])
    res['nearest_ref_dist_um'] = res['nearest_ref_dist'] * pixel
    res['nearest_ref_dist_nm'] = res['nearest_ref_dist_um'] *1000
 
        

    # Single dataframe with results:
    result = pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    return result


def get_intensities(brightest, mask_file, tiff_uuid, mask_uuid, verbose=True):
    
    directory = 'data/analysis/masks_omni/'
    file = mask_file.split('/')[-1]
    print(file)
    m_dir   = '/'.join(mask_file.split('/')[:-1])

    m = show_mask(m_dir, file, imshow=True)
    

    if verbose:
        plt.imshow(t_brightest, cmap='Blues')
        plt.imshow(brightest, cmap='Greens', alpha=0.5)
        plt.show()
    #Create a DataFrame 
    intensities = pd.DataFrame(columns=['mean_per_cell', 'area', 'intsum','intsum_px','median_in_cell', 'median_background'])

    #Loop over each cell in the mask
    for i in range(1, np.max(m)+1):
        cm = (m == i)  # Mask of the current cell
        cell_pixels = brightest[cm]  # Use pixel values of the selected cell area
        

        #Calculate statistics
        mean_per_cell = np.mean(cell_pixels)
        
        
        background = m == 0
        bgd = brightest*background
        bgd = bgd[bgd>0]
        median_background = np.median(bgd)
        
        area = cell_pixels.shape[0]
        
        intsum = np.sum(cell_pixels)
        intsum_px = intsum / area
        median_in_cell = np.median(cell_pixels)
        if median_in_cell == 0:
            print(median_in_cell)
       
       
        #Update the dataframe
        intensities.loc[i] = [mean_per_cell, area, intsum, intsum_px, median_in_cell, median_background]

    intensities['fov_mask_id'] = file
    intensities['tiff_uuid'] = tiff_uuid
    intensities['mask_uuid'] = mask_uuid
    intensities['cell_id_int'] = intensities.index
  
    return intensities





def spatial_distribution(filenames, size = 2.5, folder = "directory_path_here", feature = "Dapp_log", pixel_um=0.106):
    """
    Load per-localization/track CSVs, annotate and group tracks by experiment/cell/particle, and interactively select
    lower/upper thresholds for a per-track feature (default: Dapp_log). Cells are filtered by cell area
    (in µm^2) and tracks are limited to those within the chosen feature range. For each FOV, principal-component (PC)
    coordinates are computed for both tracks and their corresponding cell masks, saved to CSV, and used to generate
    masked 2D heatmaps in PC space and difference heatmaps across FOVs.

    Arguments:
        filenames (list[str]): Filenames (CSV) to load from `folder`, one per FOV/condition.
        size (float, default=2.5): Maximum cell area (µm^2) used to keep cells/tracks (cells with area >= size are dropped).
        folder (str): Base directory containing input CSVs and where outputs are saved.
        feature (str, default="Dapp_log"): Track-level metric used for thresholding and visualization; copied onto coords prior to filtering.
        pixel_um (float, default=0.106): Pixel size (µm/pixel) used when computing cell areas and PC-normalized coordinates.

        Run this function from console
    """
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
        coords = coords[(coords["area"]<size)]
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

    plt.ylim(-0.4, 0.4)
    plt.xlim(left = -1.1, right= 0.0)
    plt.savefig(f"{folder}/heatmap_reference.svg", dpi = 300)
    plt.show()
    plt.close()
    

    mask_contours.append(inside_img_ref)
   
    #Plot 1D localisation plot for all files and save in folder_path
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
                plt.xlim(-0.6, 0.6)
            
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
        plt.ylim(-0.4, 0.4)
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
        plt.ylim(-0.4, 0.4)
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
    plt.legend(title="State")

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
                plt.xlim(-0.6, 0.6)
            
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
        cbar.set_label('Number of Localisations')
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


 
    
