import os
import random
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit
from utils.tiff_writer import show_mask, coords_to_image 
from utils.track_trackpy import locs_preview_contours
from utils.align_mask import correct_overspill
from skimage import measure
from sklearn.decomposition import PCA
from skimage.filters import threshold_otsu
from shapely.geometry import Polygon, Point
from skimage.measure import find_contours
from utils.logging import log_and_print
from shapely.geometry import Polygon, Point
from shapelysmooth import taubin_smooth # chaikin_smooth  # Or taubin_smooth
from shapely import points, distance, boundary

import math





def get_Dapp(track_df, dT, pixel=0.106):

    '''
    this is missing frame-tolerant get_Dapp version
    from matlab pipeline:
    pixel = 0.106
    dT = 0.0107
    
    calculate D from MSD and correct for localization noise D = MSD/(4dT) -sigmaNoise^2pixel^2/dT;
    sigmaNoise is 0 for now, trackParams.pixel = 0.106
    convert into um MSD = MSD_all * pixel^2; % convert from pixel to length units
    trackParams.dT = 0.0107 
    '''

    
    sngl_track = track_df.sort_values('frame')
    
    msd = pd.DataFrame()
    msd['diff_x'] = sngl_track['x_t'].diff()
    msd['diff_y'] = sngl_track['y_t'].diff()
    msd['d_T']    = sngl_track['frame'].diff()*dT
    
    msd['dist_sq']= msd['diff_x']**2+msd['diff_y']**2
    msd['sq_d_um']= msd['dist_sq']*pixel**2
    
    Dapp = msd['sq_d_um']/(msd['d_T']*4)
    Dapp = Dapp.mean() 
    return Dapp


def meanSL(track_df):

    
    sngl_track = track_df.sort_index()
    msd = pd.DataFrame()
    msd['diff_x'] = sngl_track['x_t'].diff()
    msd['diff_y'] = sngl_track['y_t'].diff()
    msd['dist_sq']= msd['diff_x']**2+msd['diff_y']**2
    msd['dist']= np.sqrt(msd['dist_sq'])
    
    meanSL = msd['dist'].mean()
    
    return meanSL




# Diffusion fingerprint features are from PNAS 2021 Vol. 118 No. 31 e2104624118 https://doi.org/10.1073/pnas.2104624118
# Pinholt et al., Single-particle diffusional ﬁngerprinting: A machine-learning framework for quantitative analysis ofheterogeneous diffusion
# https://github.com/hatzakislab/Diffusional-Fingerprinting/blob/main/Fingerprint_feat_gen.py

def Scalings(g, dt):
# https://github.com/hatzakislab/Diffusional-Fingerprinting/blob/main/Fingerprint_feat_gen.py
    """Fit mean squared displacements to a power law.
    
    Parameters
    ----------
    msds : list-like
        mean squared displacenemts.
    
    Returns
    -------
    tuple of length 3
        The first index is the fitted generalized diffusion constant,
        the second is the scaling exponent alpha, and the final is the pvalue for the fit.
    
    """

    x=g['x_t'].to_numpy()
    y=g['y_t'].to_numpy()

    
    def msd(x, y, frac=1):
        """Computes the mean squared displacement (msd) for a trajectory (x,y) up to
        frac*len(x) of the trajectory.
    
        Parameters
        ----------
        x : list-like
            x-coordinates for the trajectory.
        y : list-like
            y-coordinates for the trajectory.
        frac : float in [0,1]
            Fraction of trajectory duration to compute msd up to.
    
        Returns
        -------
        iterable of lenght int(len(x)*frac)
            msd for the trajectory
    
        """
        N = int(len(x) * frac)
        msd = []
        for lag in range(1, N):
            msd.append(
                np.mean(
                    [
                        SquareDist(x[j], x[j + lag], y[j], y[j + lag])
                        for j in range(len(x) - lag)
                    ]
                )
            )
        return np.array(msd)

    def SquareDist(x0, x1, y0, y1):
        """Computes the squared distance between the two points (x0,y0) and (y1,y1)
    
        Returns
        -------
        float
            squared distance between the two input points
    
        """
        return (x1 - x0) ** 2 + (y1 - y0) ** 2
    

    def power(x, D, alpha):               # offset? set x0 to y0?
        return 4 * D * (x) ** alpha       # offset? set x0 to y0?

    def MSDratio(mval):
        """Computes the MSD ratio.
    
        Parameters
        ----------
        mval : list-like
            Mean squared displacements.
    
        Returns
        -------
        float
            MSD ratio.
    
        """
        return np.mean(
            [mval[i] / mval[i + 1] - (i) / (i + 1) for i in range(len(mval) - 1)]
        )

    def GetMax(x, y):
        """Computes the maximum squared distance between all points in the (x,y) set.
    
        Parameters
        ----------
        x : list-like
            x-coordinates.
        y : list-like
            y-coordinates.
    
        Returns
        -------
        float
            Largest squared distance between any two points in the set.
    
        """
        from itertools import combinations
        from random import randint
    
        A = np.array([x, y]).T
    
        def square_distance(x, y):
            return sum([(xi - yi) ** 2 for xi, yi in zip(x, y)])
    
        max_square_distance = 0
        for pair in combinations(A, 2):
            if square_distance(*pair) > max_square_distance:
                max_square_distance = square_distance(*pair)
                max_pair = pair
        return max_square_distance

    def Trappedness(x, y, maxpair, out):
        """Computes the trappedness.
    
        Parameters
        ----------
        x : list-like
            x-coordinates for the trajectory.
        y : list-like
            y-coordinates for the trajectory.
        maxpair : float
            Maximum squared pair-wise distance for the poinst in the trajectory.
        out : list-like
            Mean squared displacements.
    
        Returns
        -------
        float
            Trappedness.
    
        """
        r0 = np.sqrt(maxpair) / 2
        D = out[1] - out[0]
        return 1 - np.exp(0.2045 - 0.25117 * (D * len(x)) / r0 ** 2)

    
    msds = msd(x,y,1)
    params, pcov = curve_fit(power, np.arange(1,len(msds)+1)*dt, msds, 
                             p0=[msds[0] / (4 * dt),1], 
                             max_nfev=100000, bounds=[[0.0000001,0.],[np.inf,10]], 
                             method='trf')
    r = msds - power(np.arange(1,len(msds)+1)*dt, *params)

    params, pcov = curve_fit(power, np.arange(1,len(msds)+1)*dt, msds, sigma=np.repeat(np.std(r, ddof=1), len(msds)),
                             p0=[msds[0] / (4 * dt),1], 
                             max_nfev=100000, bounds=[[0.0000001,0.],[np.inf,10]], 
                             method='trf')

    Chival = r**2/np.var(r, ddof=1)
    Pval = stats.chi2.sf(np.sum(Chival), len(msds)-len(params))


    


    return params[0], params[1], Pval, np.mean(msds), MSDratio(msds), Trappedness(x, y, GetMax(x, y), msds)


def Efficiency(g):
# https://github.com/hatzakislab/Diffusional-Fingerprinting/blob/main/Fingerprint_feat_gen.py
    """Computes the efficiency of a trajectory, logarithm of the ratio of squared end-to-end distance
    and the sum of squared distances.

    Parameters
    ----------
    x : list-like
        x-coordinates for the trajectory.
    y : list-like
        y-coordinates for the trajectory.

    Returns
    -------
    float
        Efficiency.

    """
    x=g['x_t'].to_numpy()
    y=g['y_t'].to_numpy()

    def SquareDist(x0, x1, y0, y1):
        """Computes the squared distance between the two points (x0,y0) and (y1,y1)
    
        Returns
        -------
        float
            squared distance between the two input points
    
        """
        return (x1 - x0) ** 2 + (y1 - y0) ** 2
    
    top = SquareDist(x[0], x[-1], y[0], y[-1])
    bottom = sum(
        [SquareDist(x[i], x[i + 1], y[i], y[i + 1]) for i in range(0, len(x) - 1)]
    )
    return np.log((top) / ((len(x) - 1) * bottom))


def FractalDim(g):
# https://github.com/hatzakislab/Diffusional-Fingerprinting/blob/main/Fingerprint_feat_gen.py
    """Computes the fractal dimension using the estimator suggested by Katz & George
    in Fractals and the analysis of growth paths, 1985.

    'A simple practical method exists for classifying and comparing planar curves composed of connected line segments. This method assigns a single numberD, 
    the fractal dimension, to each curve.D =log(n)/[log(n) + log(d/L)],where:n is       the number of line segments,L is the total length of the line segments, 
    and d is the planar diameter of the curve (the greatest distance between any two endpoints). At one end of the spectrum, for straight line curves,D = 1; 
    at the other end of the spectrum, for random walk curves,D→2. Standard statistics are done on the logarithms of the fractal dimension [log(D)]. With this measure, 
    trails of biological movement, such as the growth paths of cells and the paths of wandering organisms, can be analyzed to determine the likelihood that 
    these trails are random walks and also to compare the straightness of the trails before and after experimental interventions.'

    Parameters
    ----------
    x : list-like
        x-coordinates for the trajectory.
    y : list-like
        y-coordinates for the trajectory.
    max_square_distance : float
        Maximum squared pair-wise distance for the poinst in the trajectory.

    Returns
    -------
    float
        Estimated fractal dimension.

    """
    x=g['x_t'].to_numpy()
    y=g['y_t'].to_numpy()

    def SquareDist(x0, x1, y0, y1):
        """Computes the squared distance between the two points (x0,y0) and (y1,y1)
    
        Returns
        -------
        float
            squared distance between the two input points
    
        """
        return (x1 - x0) ** 2 + (y1 - y0) ** 2
        
    def GetMax(x, y):
        """Computes the maximum squared distance between all points in the (x,y) set.
    
        Parameters
        ----------
        x : list-like
            x-coordinates.
        y : list-like
            y-coordinates.
    
        Returns
        -------
        float
            Largest squared distance between any two points in the set.
    
        """
        from itertools import combinations
        from random import randint
    
        A = np.array([x, y]).T
    
        def square_distance(x, y):
            return sum([(xi - yi) ** 2 for xi, yi in zip(x, y)])
    
        max_square_distance = 0
        for pair in combinations(A, 2):
            if square_distance(*pair) > max_square_distance:
                max_square_distance = square_distance(*pair)
                max_pair = pair
        return max_square_distance
    
    max_square_distance=GetMax(x, y)

    
    totlen = sum(
        [
            np.sqrt(SquareDist(x[i], x[i + 1], y[i], y[i + 1]))
            for i in range(0, len(x) - 1)
        ]
    )
    return np.log(len(x)) / (
        np.log(len(x)) + np.log(np.sqrt(max_square_distance) / totlen)
    )


def Gaussianity(g):
# https://github.com/hatzakislab/Diffusional-Fingerprinting/blob/main/Fingerprint_feat_gen.py
    """Computes the Gaussianity.

    Parameters
    ----------
    x : list-like
        x-coordinates for the trajectory.
    y : list-like
        y-coordinates for the trajectory.
    r2 : list-like
        Mean squared displacements for the trajectory.

    Returns
    -------
    float
        Gaussianity.

    """

    x=g['x_t'].to_numpy()
    y=g['y_t'].to_numpy()

    def QuadDist(x0, x1, y0, y1):
        return (x1 - x0) ** 4 + (y1 - y0) ** 4

    def SquareDist(x0, x1, y0, y1):
        return (x1 - x0) ** 2 + (y1 - y0) ** 2

    def msd(x, y):
    
        N = int(len(x))
        msd = []
        for lag in range(1, N):
            msd.append(
                np.mean(
                    [
                        SquareDist(x[j], x[j + lag], y[j], y[j + lag])
                        for j in range(len(x) - lag)
                    ]
                )
            )
        return np.array(msd)
    r2=msd(x, y)
    gn = []
    for lag in range(1, len(r2)):
        r4 = np.mean(
            [QuadDist(x[j], x[j + lag], y[j], y[j + lag]) for j in range(len(x) - lag)]
        )
        gn.append(r4 / (2 * r2[lag] ** 2))
    return np.mean(gn)


def Kurtosis(g):
# https://github.com/hatzakislab/Diffusional-Fingerprinting/blob/main/Fingerprint_feat_gen.py
    """Computes the kurtosis for the trajectory.

    Parameters
    ----------
    x : list-like
        x-coordinates for the trajectory.
    y : list-like
        y-coordinates for the trajectory.

    Returns
    -------
    float
        Kurtosis.

    """
    x=g['x_t'].to_numpy()
    y=g['y_t'].to_numpy()

    
    from scipy.stats import kurtosis

    val, vec = np.linalg.eig(np.cov(x, y))
    dominant = vec[:, np.argsort(val)][:, -1]
    return kurtosis([np.dot(dominant, v) for v in np.array([x, y]).T], fisher=False)


def to_PC(track_gb_item, pixel):

    track_gb_item = track_gb_item.reset_index()
    
    # print(to_PC(g, PIXEL))
    directory = os.getenv('SMPP_DATA_DIR')+'/analysis/masks_omni/'
    mask = track_gb_item['fov_mask_id'].iloc[0]

    m = show_mask(directory, mask, imshow=False)
    m = (m==track_gb_item['cell_id_int'].iloc[0])
    # display(track_gb_item)
    # plt.imshow(m)
    # plt.show()

    mask_coords = pd.DataFrame()
    mask_coords['x_t_px'] = np.where(m)[0]
    mask_coords['y_t_px'] = np.where(m)[1]

    
    pca = PCA(n_components=2)
    pca_out = pca.fit_transform(mask_coords)
    scaling = np.max(pca_out)
    pca_out = pca_out/scaling

    track_pc = pca.transform(track_gb_item[['x_t_px', 'y_t_px']])
    track_pc = track_pc/scaling

    
    # plt.scatter(pca_out[:,0], pca_out[:,1], s=1)
    # plt.scatter(track_pc[:,0], track_pc[:,1], s=1)
    # plt.axis('scaled')
    # plt.show()
    
    # fig, ax = plt.subplots(figsize=(10, 10))
    # plt.scatter(x=np.where(m)[1], y=np.where(m)[0], s=1)
    # plt.scatter(track_gb_item['y_t_px'], track_gb_item['x_t_px'], 
    #             c=track_gb_item['frame'], 
    #             s=5, 
    #             # alpha=0.7, 
    #             cmap='viridis')
    # plt.axis('scaled')

    # contours = measure.find_contours(m, 0.5)
    # for contour in contours:
    #     ax.plot(contour[:, 1], contour[:, 0], linewidth=0.5, color='r')
    # ax.axis('image')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.title('title')
    # plt.show()

    area_um_sq = np.sum(m)*pixel**2
    
    aspect = np.abs(pca_out[:,1]).max()

    track_pc0_mean_abs = np.abs(track_pc[:,0].mean())
    track_pc1_mean_abs = np.abs(track_pc[:,1].mean())

    track_pc0_sd = track_pc[:,0].std()
    track_pc1_sd = track_pc[:,1].std()
    
    
    
    return track_pc0_mean_abs, track_pc1_mean_abs, track_pc0_sd, track_pc1_sd, area_um_sq, aspect

def to_PC_raw(track_gb_item, pixel, absPC=True, flip_off=True):
    # no aggregation for plotting
    
    
    # print(to_PC(g, PIXEL))
    directory = os.getenv('SMPP_DATA_DIR')+'/analysis/masks_omni/'
    mask = track_gb_item['fov_mask_id'].iloc[0]

    m = show_mask(directory, mask, imshow=False)
    m = (m==track_gb_item['cell_id_int'].iloc[0])
    # display(track_gb_item)
    # plt.imshow(m)
    # plt.show()

    mask_coords = pd.DataFrame()
    mask_coords['x_t_px'] = np.where(m)[0]
    mask_coords['y_t_px'] = np.where(m)[1]

    
    pca = PCA(n_components=2)
    pca_out = pca.fit_transform(mask_coords)
    scaling = np.max(pca_out)
    pca_out = pca_out/scaling

    track_pc = pca.transform(track_gb_item[['x_t_px', 'y_t_px']])
    track_pc = track_pc/scaling

    
    # plt.scatter(pca_out[:,0], pca_out[:,1], s=1)
    # plt.scatter(track_pc[:,0], track_pc[:,1], s=1)
    # plt.axis('scaled')
    # plt.show()
    
    # fig, ax = plt.subplots(figsize=(10, 10))
    # plt.scatter(x=np.where(m)[1], y=np.where(m)[0], s=1)
    # plt.scatter(track_gb_item['y_t_px'], track_gb_item['x_t_px'], 
    #             c=track_gb_item['frame'], 
    #             s=5, 
    #             # alpha=0.7, 
    #             cmap='viridis')
    # plt.axis('scaled')

    # contours = measure.find_contours(m, 0.5)
    # for contour in contours:
    #     ax.plot(contour[:, 1], contour[:, 0], linewidth=0.5, color='r')
    # ax.axis('image')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.title('title')
    # plt.show()


    
    pc_df = pd.DataFrame()

    pc_df['track_pc0_raw'] = track_pc[:,0]
    pc_df['track_pc1_raw'] = track_pc[:,1]

    if flip_off:
        pc_df['track_pc0_raw'] = pc_df['track_pc0_raw']*random.choice([1, -1])
        pc_df['track_pc1_raw'] = pc_df['track_pc1_raw']*random.choice([1, -1])
    

    if absPC:
        pc_df['track_pc0_raw_abs'] = np.abs(track_pc[:,0])
        pc_df['track_pc1_raw_abs'] = np.abs(track_pc[:,1])
        pc_df.drop(['track_pc0_raw', 'track_pc1_raw'], axis=1, inplace=True)
    
    
    
    return pc_df


def get_ints_by_track(image_2ch, meta_2ch, tracking_df, tracks_2ch):
# Get intensity values in channel 2, separate value for each localization in track
    
    tracking_df = tracking_df.xs(meta_2ch[3], level=0, drop_level=False)
    
    # getting mask
    directory = os.getenv('SMPP_DATA_DIR')+'/analysis/masks_omni/'
    file = meta_2ch[2].split('/')[-1]
    print(file)
    m = show_mask(directory, file, imshow=False)

    fluo_2ch_proj = np.mean(image_2ch, axis=0)
    
    tracking_df['x_raw_rounded'] = tracking_df['x'].astype('int') # no .round(0) here! 
    tracking_df['y_raw_rounded'] = tracking_df['y'].astype('int') # no .round(0) here! 
    
    tracking_df['intensity_2ch_by_track']=fluo_2ch_proj[tracking_df['x_raw_rounded'], 
        tracking_df['y_raw_rounded']]

    # plt.imshow(fluo_2ch_proj)
    locs_preview_contours(tracking_df['y_t'], tracking_df['x_t'], tracking_df['intensity_2ch_by_track'], m, 'intensity ch2')
    
    tracks_2ch.append(tracking_df)


def get_area(track_gb_item, pixel):
    '''
    returns area in um2
    for cases when PC is not needed, just area
    '''
    directory = os.getenv('SMPP_DATA_DIR')+'/analysis/masks_omni/'
    mask = track_gb_item['fov_mask_id'].iloc[0]

    m = show_mask(directory, mask, imshow=False)
    m = (m==track_gb_item['cell_id_int'].iloc[0])
    area_um_sq = np.sum(m)*pixel**2
    
    return area_um_sq


def get_norm_ints_2ch(fluo, fluos_list, masks_list, PC_scaling = True, overspill_correction = True, lower=0, upper=42, pixel_size=0.106, flip_off=True):
    '''
    fluo:           2nd channel data as pandas df
    fluos_list:     list to populate with normalized and PC-transformed 2nd channel data
    masks_list:     same as above, for masks 
    lower:          lower bound for cell area to take, squared micron
    upper:          upper bound for cell area to take, squared micron
    pixel_size:     xy pixel size in micron
    '''

    print(f'Cells with area in [{lower}, {upper}] square microns proceed to analysis')
    
    lower, upper = (lower/pixel_size**2, upper/pixel_size**2) # convert to pixel units
    
    # get mask
    directory = os.getenv('SMPP_DATA_DIR')+'/analysis/masks_omni/'
    file = fluo[2].split('/')[-1]
    m = show_mask(directory, file, imshow=False)


    # remove everything outside the masks
    cells = []
    masks = []
    
    for i in range(1, np.max(m)+1):    # all cells

        cm = (m==i) # cell mask

        if np.sum(cm)>lower and np.sum(cm)<upper: # cell area range

            mask_coords = pd.DataFrame()
            mask_coords['x_t'] = np.where(cm)[0]
            mask_coords['y_t'] = np.where(cm)[1]
            
            # get locs for mask
            in_cell=fluo[0].copy()
            
            # # causing systematic bias?
            # in_cell['x_rounded']=in_cell['x_t'].astype('int') # no .round(0) here! 
            # in_cell['y_rounded']=in_cell['y_t'].astype('int')

            in_cell['x_rounded'] = np.round(in_cell['x_t']).astype(int)
            in_cell['y_rounded'] = np.round(in_cell['y_t']).astype(int)
            
            in_cell = in_cell.loc[((in_cell['x_rounded']<cm.shape[0])&(0<in_cell['x_rounded']))&\
                                  ((in_cell['y_rounded']<cm.shape[1])&(0<in_cell['y_rounded']))]
            in_cell['in_cell']=cm[in_cell['x_rounded'], in_cell['y_rounded']]
            in_cell = in_cell.loc[in_cell['in_cell']==True]

            
            # cell index
            in_cell['cell_id_int']=i
            mask_coords['cell_id_int']=i
            
            # overspill correction 
            if overspill_correction:
                in_cell = correct_overspill(in_cell)

                # normalize corrected
                in_cell['value_corr_norm']=in_cell['value_corr']-in_cell['value_corr'].min()
                in_cell['value_corr_norm']=in_cell['value_corr_norm']/(in_cell['value_corr_norm'].max())
    
                # normalize correccted, where still bad - drop whole cell
                in_cell['value_corr_norm_f']=in_cell['value_corr_filtered']-in_cell['value_corr_filtered'].min()
                in_cell['value_corr_norm_f']=in_cell['value_corr_norm_f']/(in_cell['value_corr_norm_f'].max())
                    
            # normalize
            in_cell['value_norm']=in_cell['value']-in_cell['value'].min()
            in_cell['value_norm']=in_cell['value_norm']/(in_cell['value_norm'].max())

        
            # PC transformed coords
            pca = PCA(n_components=2)
            pca.fit(mask_coords[['x_t', 'y_t']])                        # fit on mask
            pca_out = pca.transform(in_cell[['x_t', 'y_t']])            # transform second channel
            pca_cm_coords = pca.transform(mask_coords[['x_t', 'y_t']])  # transform mask

            if PC_scaling:
                scaling = np.max(np.abs(pca_cm_coords))
                pca_out = pca_out/scaling
                pca_cm_coords = pca_cm_coords/scaling
            in_cell['pc0_raw'] = pca_out[:,0]
            in_cell['pc1_raw'] = pca_out[:,1]
        
            mask_coords['pc0_raw'] = pca_cm_coords[:,0]
            mask_coords['pc1_raw'] = pca_cm_coords[:,1]
            
            if flip_off:
                in_cell['pc0_raw'] = in_cell['pc0_raw']*random.choice([1, -1])
                in_cell['pc1_raw'] = in_cell['pc1_raw']*random.choice([1, -1])
                
                mask_coords['pc0_raw'] = mask_coords['pc0_raw']*random.choice([1, -1])
                mask_coords['pc1_raw'] = mask_coords['pc1_raw']*random.choice([1, -1])

                
            # add results to list of cells
            cells.append(in_cell)
            masks.append(mask_coords)

    # all data into a single df
    all_cells = pd.concat(cells)
    all_cells['fov_mask_id'] = file
    fluos_list.append(all_cells)

    all_masks = pd.concat(masks)
    all_masks['fov_mask_id'] = file
    masks_list.append(all_masks)



def get_ints_bt_norm(image_2ch_norm: np.ndarray, 
                     meta_2ch: tuple, 
                     tracks_2ch_fov: pd.DataFrame) -> pd.DataFrame:
    
    '''
    # Get intensity values in channel 2, separate value for each localization in track
    # image_2ch_norm is 2d array, already projected, pixel resolution, intensity normalized by cell
    # meta_2ch is tuple from master_im_data_2ch 
        
    # 'i_2ch_bt_norm_bc' :
    # intensity in 2nd channel, normalized by cell, overspill corrected. 
    # A value for each loc, to be aggregated later by track
    
    
    '''
    
    # getting mask
    directory = os.getenv('SMPP_DATA_DIR')+'/analysis/masks_omni/'
    file = meta_2ch[2].split('/')[-1]
    print(file)
    m = show_mask(directory, file, imshow=False)
  
    # Bounds check
    valid_mask = ((tracks_2ch_fov['x_rounded'] >= 0) & 
                  (tracks_2ch_fov['x_rounded'] < image_2ch_norm.shape[0]) &
                  (tracks_2ch_fov['y_rounded'] >= 0) & 
                  (tracks_2ch_fov['y_rounded'] < image_2ch_norm.shape[1]))
    
    tracks_2ch_fov.loc[valid_mask, 'i_2ch_bt_norm_bc'] = image_2ch_norm[
        tracks_2ch_fov.loc[valid_mask, 'x_rounded'].values, 
        tracks_2ch_fov.loc[valid_mask, 'y_rounded'].values
    ]

    # plt.imshow(fluo_2ch_proj)
    locs_preview_contours(tracks_2ch_fov['y_t'], 
                          tracks_2ch_fov['x_t'], 
                          tracks_2ch_fov['i_2ch_bt_norm_bc'], 
                          m, 
                          'norm fltrd intensity ch2')
    
    return tracks_2ch_fov




def get_ints_bt_norm_Otsu(image_2ch_norm: np.ndarray,
                          meta_2ch: tuple,
                          df_tracks_2ch_norm: pd.DataFrame,
                         ) -> pd.DataFrame:
    
    

    df_tracks_2ch_norm = df_tracks_2ch_norm.reset_index()
    
    # Step 1: Extract non-NaN values for Otsu threshold calculation
    valid_pixels = image_2ch_norm[~np.isnan(image_2ch_norm)]
    
    # Compute Otsu threshold
    otsu_threshold = threshold_otsu(valid_pixels)
    log_and_print(f"Otsu threshold value: {otsu_threshold}")
    
    # Step 2: Fill NaNs with zeros for contour finding
    image_filled = np.nan_to_num(image_2ch_norm, nan=0.0)
    
    # Step 3: Create binary mask (values above threshold)
    binary_mask = image_filled > otsu_threshold
    
    # Step 4: Find contours at the threshold level
    # find_contours uses marching squares algorithm
    contours = find_contours(image_filled, otsu_threshold)
    
    # Step 5: Convert contours to Shapely polygons
    polygons = []
    for contour in contours:
        # find_contours returns (row, col) coordinates, we may need to swap to (x, y)
        # Also ensure we have at least 3 points to form a valid polygon
        if len(contour) >= 3:
            try:
                # swap row/col to x/y if needed (depends on your coordinate system)
                polygon = Polygon(contour)
                
                # Validate polygon
                if polygon.is_valid:
                    polygons.append(polygon)
                else:
                    # Try to fix invalid polygon
                    polygon = polygon.buffer(0)
                    if polygon.is_valid:
                        polygons.append(polygon)
            except Exception as e:
                log_and_print(f"Warning: Could not create polygon from contour: {e}")
                continue
    
    log_and_print(f"Found {len(polygons)} valid polygons from threshold isolines")
    
    # Step 6: Calculate distance from each point to nearest polygon
    # Only for rows where i_2ch_bt_norm_bc is not NaN
    
    distances = []
    
    for idx, row in df_tracks_2ch_norm.iterrows():
        x, y = row['x_t'], row['y_t']
        point = Point(x, y)
        
        # Calculate distance to each polygon
        min_distance = float('inf')
        
        for polygon in polygons:
            if polygon.contains(point):
                # Point is inside a polygon - distance is 0
                min_distance = 0.0
                break
            else:
                # Point is outside - calculate distance to boundary
                dist = point.distance(polygon)
                min_distance = min(min_distance, dist)
        
        # Handle case with no polygons found
        if min_distance == float('inf'):
            min_distance = np.nan
        
        distances.append(min_distance)


    df_tracks_2ch_norm['dist_2ch_polygon'] = distances
    
    mask = df_tracks_2ch_norm['i_2ch_bt_norm_bc'].isna()
    df_tracks_2ch_norm.loc[mask, 'dist_2ch_polygon']=np.nan
    
    
    return df_tracks_2ch_norm





# Function to getting angle


def generate_angle_series(s):
    def Get_angle_x(g):
        x=g['x_t'].to_numpy()
        return x
    def Get_angle_y(g):
        y=g['y_t'].to_numpy()
        return y
    def angle_between_three_points(A, B, C, D, E, F):
        # Compute vectors
        v1=(C-A,D-B)
        v2=(E-C,F-D)
    
        dot=v1[0]*v2[0]+v1[1]*v2[1]
    
        norm1=math.hypot(v1[0],v1[1])
        norm2=math.hypot(v2[0],v2[1])
    
        if norm1==0 or norm2==0:
            return None  
    
        cos_theta=dot/(norm1*norm2)
        # Clamping for numerical stability
        cos_theta=max(-1.0,min(1.0,cos_theta))
    
        angle_rad=math.acos(cos_theta)
        angle_deg=math.degrees(angle_rad)
        return angle_deg
    angle_test_x=s.apply(eval('Get_angle_x'), include_groups=False)
    angle_test_y=s.apply(eval('Get_angle_y'), include_groups=False)
    angle_series=pd.Series(index=angle_test_x.index, dtype=object) 
    for label in angle_series.index:
        angle_series.loc[label] = np.empty(0, dtype=float)
    for particle_i in range(angle_test_x.shape[0]):
        for location_j in range(1, angle_test_x.iloc[particle_i].shape[0]-1):
            new_angle = np.float64(angle_between_three_points(angle_test_x.iloc[particle_i][location_j-1],angle_test_y.iloc[particle_i][location_j-1],angle_test_x.iloc[particle_i][location_j], angle_test_y.iloc[particle_i][location_j],angle_test_x.iloc[particle_i][location_j+1], angle_test_y.iloc[particle_i][location_j+1]))
            angle_series.iloc[particle_i]=np.append(angle_series.iloc[particle_i], new_angle)

    Fitting_angle_array=pd.Series(index=angle_series.index, dtype=object) 
    Fitting_angle_array_mean=pd.Series(index=angle_series.index, dtype=object) 
    Fitting_angle_array_CV=pd.Series(index=angle_series.index, dtype=object) 
    for particle_i in range(angle_series.shape[0]):
        #Fitting_angle_array.iloc[particle_i]=angle_series.iloc[particle_i]
        Fitting_angle_array_mean.iloc[particle_i]=np.mean((angle_series.iloc[particle_i]))
        Fitting_angle_array_CV.iloc[particle_i]=(np.median((angle_series.iloc[particle_i]))/np.mean((angle_series.iloc[particle_i])))
    return Fitting_angle_array_mean, Fitting_angle_array_CV


def compute_MBR(df, t=1, tau=1, l=0.0001):
    # see Knotz, Kruger, 2024, https://doi.org/10.1103/PhysRevE.110.044137
    # t and tau are in steps, l is in micron
    
    positions_x = df['x_t'].to_numpy()
    positions_y = df['y_t'].to_numpy()

    Back_relaxation = []
    for i in range(tau, positions_x.shape[0]-t):
        d_past_x=positions_x[i]-positions_x[i-tau]
        d_past_y=positions_y[i]-positions_y[i-tau]
        d_future_x=positions_x[i+t]-positions_x[i]
        d_future_y=positions_y[i+t]-positions_y[i]
        
        d_square=d_past_x**2+d_past_y**2 # past displacement
        
        b_doc_product=d_past_x*d_future_x+d_past_y*d_future_y# future displacement
        if d_square < l**2:
            continue
        else:
        # avoid division by zero (though cutoff should make d != 0 in practice)
            current_BR =(b_doc_product/d_square)*(-1)
            Back_relaxation.append(current_BR)
    
    if len(Back_relaxation) == 0:
        raise ValueError('No valid triplets — cutoff l too large or trajectory too short.')
    
    mbr = np.mean(Back_relaxation)
    return mbr



def get_dist_to_edge(df    : pd.DataFrame,   # data
                     masks : pd.DataFrame,   # data_masks
                     ) -> pd.DataFrame:

    
    
    '''
    returns distance to cell edge (as polygon) in pixels, aggregated by track
    
    data_masks
    columns: Index(['Unnamed: 0', 
        'x_t', 'y_t', 'cell_id_int', 'pc0_raw', 'pc1_raw',
        'fov_mask_id', 'experiment', 'rep_label', 'mask_uuid'],
        dtype='object') 

    data    
    columns: Index(['Unnamed: 0', 'mask_uuid', 'cell_id_int', 'particle', 'x', 'y', 'mass',
       'size_y', 'size_x', 'ecc', 'signal', 'raw_mass', 'ep', 'frame', 'x_t',
       'y_t', 'x_rounded', 'y_rounded', 'in_cell', 'fov_mask_id', 'tiff_uuid',
       'comment', 'track_length', 'Dapp', 'Dapp_log', 'x_raw_rounded',
       'y_raw_rounded', 'intensity_2ch_by_track', 'i_2ch_bt_norm_bc',
       'dist_2ch_polygon', 'experiment', 'fov_uids', 'rep_label', 'strain',
       'x_t_px', 'y_t_px'],
      dtype='object') 
    
    '''

    df = df.copy()
    for mask_id in df['mask_uuid'].unique():
        # print(mask_id)
        mask_df = masks[masks['mask_uuid']==mask_id]
        df_m = df[df['mask_uuid']==mask_id]
        # plt.scatter(mask_df['x_t'], mask_df['y_t'], c=mask_df['cell_id_int'])
        im = coords_to_image(mask_df.rename(columns={'x_t':'x_rounded', 'y_t':'y_rounded'}),
                             'cell_id_int')
        # plt.imshow(im)
        # plt.show()

        polygons = []
        # get contour as polygon
        # plt.figure(figsize=(20, 10))
        for cell_id in mask_df['cell_id_int'].unique():
            # print(cell_id)
            contour = find_contours(im==cell_id, 0.5)
            polygon = Polygon(contour[0])
            smooth_polygon = taubin_smooth(polygon, factor=0.75, mu=-0.6, steps=7)
            polygons.append(smooth_polygon)
            x, y = smooth_polygon.exterior.xy
            # plt.plot(x, y)

            locs = df_m[df_m['cell_id_int']==cell_id]
            pts = points(
                            locs["x_t_px"].to_numpy(),
                            locs["y_t_px"].to_numpy()
                        )

            d = distance(pts, boundary(smooth_polygon))
            # plt.scatter(locs['x_t_px'], locs['y_t_px'], c=d)
            
            df.loc[locs.index, 'dist_to_edge_px'] = d
        
            

        # plt.axis('equal')            
        # plt.show()

    df.set_index(['experiment', 'mask_uuid', 'cell_id_int', 'particle'], inplace=True)

    # return df # if you need all locs data 
    return df['dist_to_edge_px'].groupby(df.index).apply('mean')




