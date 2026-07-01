import tifffile
from io import BytesIO
import trackpy as tp
from matplotlib import pyplot as plt
import numpy as np
from utils.logging import log_args_and_time



from utils.tp_refine_com_arr_monkey_patch import refine_com_arr_mp, refine_com_mp
tp.refine.center_of_mass.refine_com_arr = refine_com_arr_mp
tp.refine.center_of_mass.refine_com = refine_com_mp
tp.refine.refine_com = refine_com_mp
# inspect.getsource(tp.refine.center_of_mass.refine_com_arr) # to check code used when calling func


'''
based partly on trackpy 
https://github.com/soft-matter/trackpy/blob/master/trackpy/feature.py


locate(raw_image, diameter, minmass=None, maxsize=None, separation=None,
           noise_size=1, smoothing_size=None, threshold=None, invert=False,
           percentile=64, topn=None, preprocess=True, max_iterations=10,
           filter_before=None, filter_after=None,
           characterize=True, engine='auto')
'''







# def get_locs_trackpy(fluo_from_allas, threshold, tp_d=11, verbose = True, ): 

@log_args_and_time
def get_locs_trackpy(fluo_from_allas, minmass_threshold, bandpass_threshold, tp_d=11, gauss_short=1, boxcar_long=7, verbose=True, h5=False): 
    
    tp.quiet(suppress=True)

    if h5:
        image = fluo_from_allas
        first_frame = 0    # before h5, widefield was stored as 1 frame of tiff

    else:
        image = tifffile.imread(BytesIO(fluo_from_allas))
        first_frame = 1
    
    # used for mask registration 
    proj_sd = np.std(image[first_frame:,:,:], axis = 0)

    plt.imshow(image[0])
    plt.show()
    print('data loaded, processing')
    
    rand_frames = np.random.randint(500, size=3).tolist()
    if minmass_threshold == 'auto':
        
        ts = []
        for rframe in rand_frames:
            f = tp.locate(image[rframe], tp_d, minmass=20)
            t = f['mass'].quantile(0.90)
            ts.append(t)
            print(t)

        minmass_threshold = np.mean(ts)
        print('threshold used: '+str(minmass_threshold))

    data = tp.batch(image, 11, minmass=float(minmass_threshold), noise_size=gauss_short, smoothing_size=boxcar_long, threshold=int(bandpass_threshold))
    data.rename(columns={'x': 'y', 'y': 'x'}, inplace=True)        
    
    # for rframe in rand_frames:
    #     f = tp.locate(image[rframe], 11, minmass=float(minmass_threshold), noise_size=gauss_short, smoothing_size=boxcar_long, threshold=int(bandpass_threshold))
    #     tp.annotate(f, image[rframe])
    #     plt.show()
    
    return data, proj_sd


def get_locs_preview(fluo_from_allas, minmass_threshold, bandpass_threshold, prev_frame, tp_d=11, gauss_short=1, boxcar_long=7, verbose=True): 
    
    tp.quiet(suppress=True)
    image = tifffile.imread(BytesIO(fluo_from_allas))
    

    f = tp.locate(image[prev_frame], 11, minmass=float(minmass_threshold), noise_size=gauss_short, smoothing_size=boxcar_long, threshold=int(bandpass_threshold))
    tp.annotate(f, image[prev_frame])
    plt.show()
    
    return 

def h5_get_locs_preview(fluo_from_allas, minmass_threshold, bandpass_threshold, prev_frame, tp_d=11, gauss_short=1, boxcar_long=7, verbose=True): 
    
    tp.quiet(suppress=True)
 
    f = tp.locate(fluo_from_allas[prev_frame], 11, minmass=float(minmass_threshold), noise_size=gauss_short, smoothing_size=boxcar_long, threshold=int(bandpass_threshold))
    tp.annotate(f, fluo_from_allas[prev_frame])
    plt.show()
    
    return 


