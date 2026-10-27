import os
import numpy as np
import ipywidgets
import seaborn as sns



from skimage.filters import gaussian, threshold_multiotsu
from skimage.morphology import binary_erosion, remove_small_objects, remove_small_holes, disk
from skimage import img_as_float, measure
from scipy import ndimage
from pystackreg import StackReg
from ipywidgets import interactive, Layout
from matplotlib import pyplot as plt
from utils.tiff_writer import show_mask




def align_mask(fluo_sd_proj, mask_array, tform, verbose=True):


    mask_array = mask_array>0
    mask_array = mask_array.astype('float64')



    def binary(dapi, gauss=1, classes=3, nbins=1000, min_size=100, area_threshold=100):
        dapi_g = img_as_float(gaussian(dapi, gauss))
        thresholds = threshold_multiotsu(dapi_g, classes=classes, nbins=nbins)
        mask_general = dapi_g > thresholds[0]
        # clean_holes = remove_small_objects(mask_general, min_size=min_size)
        # clean = remove_small_holes(clean_holes, area_threshold=area_threshold)
        return mask_general

    bin_proj = binary(fluo_sd_proj)
    norm_proj = fluo_sd_proj-np.min(fluo_sd_proj)
    norm_proj = norm_proj/np.max(norm_proj)

    if tform == 'RIGID_BODY':
        sr = StackReg(StackReg.RIGID_BODY)
    elif tform == 'TRANSLATION':
        sr = StackReg(StackReg.TRANSLATION)
    else:
        print('unknown transformation')

    tmats = out_tra = sr.register(norm_proj, mask_array)
    
    
    if verbose:
        plt.imshow(fluo_sd_proj)
        plt.title('SD projection')
        plt.show()
        
        plt.imshow(bin_proj)
        plt.title('binarized fluo (for demonstration only)')
        plt.show()
      
        plt.imshow(bin_proj, cmap='Blues')
        plt.imshow(mask_array, cmap='Greens', alpha=0.5)
        plt.title('before registration')
        plt.show()

        plt.imshow(bin_proj, cmap='Blues')
        plt.imshow(sr.transform(mask_array), cmap='Greens', alpha=0.5)
        plt.title('after registration ' + tform)
        plt.show()
        
        print('transformations:')
        print(tmats)
    return (tmats)

def manual_reg_plot(xlim, ylim, xshift, yshift, fov_n, fovs_list, offsets_tuple, fluo=False):

    offsets_dict = offsets_tuple[0]
    
    all_fovs_locs = fovs_list
    directory = os.getenv('SMPP_DATA_DIR')+'/analysis/masks_omni/'
    file = all_fovs_locs[fov_n][2].split('/')[-1]
    print(file)
    c = show_mask(directory, file, imshow=False)

    fig, ax = plt.subplots(figsize=(8, 8))

    locs = all_fovs_locs[fov_n][0]

    # alignment of fluos needs color
    if fluo: 
        sns.scatterplot(x = locs['y'] + 0.2*xshift, 
                        y = locs['x'] + 0.2*yshift, 
                        hue=locs['value'], 
                        s=3, 
                        # alpha=0.3, 
                        palette='viridis',
                        legend=False,
                        edgecolor='none',
                   )

    else: 
        plt.scatter(locs['y'] + 0.2*xshift, locs['x'] + 0.2*yshift, 
                        # c=color, 
                        s=1, 
                        alpha=0.3, 
                        # cmap='viridis',
                   )

    
    
    # plt.imshow(c, alpha=0.3, cmap='prism')
    plt.axis('scaled')
    

    for i in range(np.max(c)):
        cell = (c==i)
        contours = measure.find_contours(cell, 0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=0.5, color='r')
        ax.axis('image')
        # ax.set_xticks([])
        # ax.set_yticks([])

    plt.xlim(xlim)
    plt.ylim(ylim)
    
    plt.show()

    
    button = ipywidgets.Button(description="Save offset")
    output = ipywidgets.Output()
    
    display(button, output)
    
    def on_button_clicked(b):
        with output:
            # print(interactive_plot.children[2].value)
            # print(interactive_plot.children[3].value)
            locs['x_t'] = locs['x'] + 0.2*yshift 
            locs['y_t'] = locs['y'] + 0.2*xshift
            offsets_dict[file] = (0.2*xshift, 0.2*yshift)
            
            print(f'offset for FOV {file} is set to: {str((0.2*xshift, 0.2*yshift))}')
    
    button.on_click(on_button_clicked)


# mplt((20, 40), (10, 25), 10, 10, 0)


def correct_overspill(cell_df, dist_threshold_quantiles = (0.3, 0.6), intensity_threshold_quantile = 0.6): # decent for nucleoid: dist_threshold_quantiles = (0.3, 0.6), intensity_threshold_quantile = 0.6
    '''
    dist_threshold_quantile:      tuple, 
                                  [0] defines the thickness of the cell rim to check for overspill
                                  [1] delete whole cell if pixel here is still above intensity_threshold
                                  
    intensity_threshold_quantile: quantile from intensity in inner part to distinguish overspill from healthy pixel 

    Returns cell_df dataframe, with added columns 'value_corr' and 'value_corr_filtered'
    
    '''
    def get_distance_map(df):
        # df = cell data, must have ['x_t', 'y_t']
        
        x_unique = np.sort(df['x_t'].unique())
        y_unique = np.sort(df['y_t'].unique())
        grid = np.zeros((len(y_unique), len(x_unique)), dtype=bool)
        
        x_idx = {v: i for i, v in enumerate(x_unique)}
        y_idx = {v: i for i, v in enumerate(y_unique)}
    
        for _, row in df.iterrows():
            grid[y_idx[row['y_t']], x_idx[row['x_t']]] = row['value'] > 0
    
        grid_padded = np.pad(grid, pad_width=1, mode='constant', constant_values=0)
        dist_padded = ndimage.distance_transform_edt(grid_padded)
        dist = dist_padded[1:-1, 1:-1]
        dist_flat = []
        for _, row in df.iterrows():
            dist_flat.append(dist[y_idx[row['y_t']], x_idx[row['x_t']]])

        return dist_flat

        
    def plot_cell_overspill_debug(df_with_values, color=('d_edge', 'check1'), title=''):
        # plot the cell for debugging 
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        sns.scatterplot(data=df_with_values, x='x_t', y='y_t', hue=color[0], palette='Grays', legend=False, s=42*4, ax=axes[0])
        sns.scatterplot(data=df_with_values, x='x_t', y='y_t', hue=color[1], palette='Blues', legend=False, s=42*4, ax=axes[1])
            
        plt.title(title)
        plt.tight_layout()
        plt.show()
        plt.close()

        
    cell_df['d_edge'] = get_distance_map(cell_df)
    cell_df['value_corr'] = cell_df['value'].copy()
    

    pxls_deep   = cell_df['d_edge'].max() * dist_threshold_quantiles[0]
    pxls_deeper = cell_df['d_edge'].max() * dist_threshold_quantiles[1]
    
    # distance threshold to define the rim
    cell_df['check0'] = (cell_df['d_edge'] < pxls_deep)
    cell_df['check0'] = cell_df['check0'].astype(int)

    # pixels between dist_threshold_quantiles[0] and [1]
    cell_df['check1'] = ((cell_df['d_edge'] >= pxls_deep) & 
                      (cell_df['d_edge'] < pxls_deeper))
    cell_df['check1'] = cell_df['check1'].astype(int)

    

    ''' correction STEP0 '''
    
    # plot_cell_overspill_debug(cell_df, title='before corr', color=('value', 'check1'))
    # np.nan where intensity is higher than intensity_threshold
    cell_df.loc[(cell_df['check0'])&(cell_df['value']>intensity_threshold_quantile*cell_df['value'].max()), 'value_corr']=np.nan
    # plot_cell_overspill_debug(cell_df, title='after corr', color=('value', 'check1'))


    ''' correction STEP1 '''
    
    cell_df['value_corr_filtered'] = cell_df['value_corr'].copy()
    if cell_df.loc[cell_df['check1'].astype(bool), 'value_corr'].max()==cell_df['value_corr'].max():
        cell_df.loc[:, 'value_corr_filtered']=np.nan
    

    
    
    
    
    return cell_df







