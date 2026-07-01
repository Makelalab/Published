import os
import numpy as np
import ipywidgets

from skimage.filters import gaussian, threshold_multiotsu
from skimage.morphology import binary_erosion, remove_small_objects, remove_small_holes, disk
from skimage import img_as_float, measure
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

def manual_reg_plot(xlim, ylim, xshift, yshift, fov_n, fovs_list, offsets_tuple):

    offsets_dict = offsets_tuple[0]
    
    all_fovs_locs = fovs_list
    directory = os.getenv('SMPP_DATA_DIR')+'/analysis/masks_omni/'
    file = all_fovs_locs[fov_n][2].split('/')[-1]
    print(file)
    c = show_mask(directory, file, imshow=False)

    fig, ax = plt.subplots(figsize=(8, 8))

    locs = all_fovs_locs[fov_n][0]
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
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='r')
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
