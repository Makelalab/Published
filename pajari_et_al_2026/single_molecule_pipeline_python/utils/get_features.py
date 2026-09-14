import os
import pandas as pd
import numpy as np
import random
import scipy.stats as stats
from scipy.optimize import curve_fit
from utils.tiff_writer import show_mask 
from utils.track_trackpy import locs_preview_contours
from skimage import measure
from sklearn.decomposition import PCA




def get_Dapp(track_df, dT, pixel=0.106):

    # from matlab pipeline:
    # pixel = 0.106
    # dT = 0.0107
    
    # calculate D from MSD and correct for localization noise D = MSD/(4dT) -sigmaNoise^2pixel^2/dT;
    # sigmaNoise is 0 for now, trackParams.pixel = 0.106
    # convert into um MSD = MSD_all * pixel^2; % convert from pixel to length units
    # trackParams.dT = 0.0107 


    
    sngl_track = track_df.sort_index()
    msd = pd.DataFrame()
    msd['diff_x'] = sngl_track['x_t'].diff()
    msd['diff_y'] = sngl_track['y_t'].diff()
    msd['dist_sq']= msd['diff_x']**2+msd['diff_y']**2
    msd['sq_d_um']= msd['dist_sq']*pixel**2
    
    Dapp = msd['sq_d_um'].mean()/(4*float(dT))
        
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

        
    # Create an empty DataFrame to store all mask coordinates
    mask_coords = pd.DataFrame()
      

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

    if absPC:
        pc_df['track_pc0_raw_abs'] = np.abs(track_pc[:,0])
        pc_df['track_pc1_raw_abs'] = np.abs(track_pc[:,1])
        pc_df.drop(['track_pc0_raw', 'track_pc1_raw'], axis=1, inplace=True)
        
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






    