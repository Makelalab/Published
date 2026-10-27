'''
This is based on DECODE by TuragaLab, see
https://github.com/TuragaLab/DECODE
'''

import torch
import numpy as np
import pandas as pd
import PIL

import os
import sys
import yaml
from pathlib import Path
# from pprint import pprint
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import decode
import decode.neuralfitter.train
import decode.utils

from decode.utils import param_io
import tifffile
from io import BytesIO
import copy


def get_locs_decode(fluo_from_allas, verbose = True, prob = 1):
        
    param_file = 'data/models_decode/decode_20240226/param_run.yaml' # not configured for now
    param = param_io.load_params(param_file)
    model_file = 'data/models_decode/decode_20240226/model_2.pt' #@param {type:"string"} # not configured for now

    model = decode.neuralfitter.models.SigmaMUNet.parse(param)
    model = decode.utils.model_io.LoadSaveModel(model,
                                                input_file=model_file,
                                                output_file=None).load_init()

    # to read directly from buffer as here
    # https://github.com/cgohlke/tifffile/issues/55
    
    image = tifffile.imread(BytesIO(fluo_from_allas)) 

    # used for mask registration 
    proj_sd = np.std(image[1:,:,:], axis = 0)
    
    # # this works, and we can extract the timestamps when necessary:
    # tif = tifffile.TiffFile(BytesIO(ttff))
    # for i in tif.pages[0].tags:
    #     print('tag: {}, value: {}'.format(i.name, i.value))

    
    plt.imshow(image[0])
    plt.show()
    print('data loaded, processing')

    frames = torch.from_numpy(image.astype('float32'))
    
    mirror_frame = True #@param {type:"boolean"}
    camera = decode.simulation.camera.Photon2Camera.parse(param)
    camera.em_gain = 50
    camera.device = 'cpu'
    device = 'cuda:0'
    
    # post-processing
    post_raw_th = 0.1
    frame_proc = decode.neuralfitter.utils.processing.TransformSequence([
        decode.neuralfitter.utils.processing.wrap_callable(camera.backward),
        decode.neuralfitter.frame_processing.AutoCenterCrop(8),
        decode.neuralfitter.frame_processing.Mirror2D(dims=-1) if mirror_frame else [],
        decode.neuralfitter.scale_transform.AmplitudeRescale.parse(param)
    ])
    
    # determine extent of frame and its dimension after frame_processing
    size_procced = decode.neuralfitter.frame_processing.get_frame_extent(frames.unsqueeze(1).size(), frame_proc.forward)  
    # frame size after processing
    frame_extent = ((-0.5, size_procced[-2] - 0.5), (-0.5, size_procced[-1] - 0.5))
    
    # Setup post-processing
    # It's a sequence of backscaling, relative to abs. coord conversion and frame2emitter conversion
    post_proc = decode.neuralfitter.utils.processing.TransformSequence([
    
        decode.neuralfitter.scale_transform.InverseParamListRescale.parse(param),
    
        decode.neuralfitter.coord_transform.Offset2Coordinate(xextent=frame_extent[0],
                                                              yextent=frame_extent[1],
                                                              img_shape=size_procced[-2:]),
    
        decode.neuralfitter.post_processing.SpatialIntegration(raw_th=0.1,
                                                               xy_unit='px',
                                                               px_size=param.Camera.px_size)
    
    
    ])
    
    print("Decode pipeline successfully set up.")

    infer = decode.neuralfitter.Infer(model=model, ch_in=param.HyperParameter.channels_in,
                                      frame_proc=frame_proc, post_proc=post_proc,
                                      device=device, num_workers=0, pin_memory=False,
                                      batch_size='auto')
    
    emitter = infer.forward(frames)

    if verbose:
        
        print(emitter)
        
        random_ix = torch.randint(frames.size(0), size=(1, )).item()
        em_subset = emitter.get_subset_frame(random_ix, random_ix)
        plt.figure(figsize=(8, 8))
        decode.plot.PlotFrameCoord(frame=frame_proc.forward(frames[[random_ix]]),
                                   pos_out=em_subset.xyz_px, phot_out=em_subset.prob).plot()
        plt.title('random frame')
        plt.show()
        
        #@markdown > Check distribution of detection probability and Sigma estimate
        plt.figure(figsize=(12, 4))
        
        plt.subplot(121)
        # plt.hist(emitter.prob, bins=50)
        sns.distplot(emitter.prob, bins=50, norm_hist=True, kde=False)
        plt.xlabel(r'$p$')
        plt.ylabel('rel. frequency')
        plt.title('Detection Probability')
        
        plt.subplot(122)
        # plt.hist(emitter.xyz_sig_nm[:, 0], bins=50)
        sns.distplot(emitter.xyz_sig_nm[:, 0], bins=50, norm_hist=True, kde=True)
        plt.xlabel(r'$\sigma_x$ [nm]')
        plt.ylabel('rel. frequency')
        plt.title(r'Sigma Estimate in $x$')
        
        plt.show()
    
        print("""Here we compare the inferred distribution of the photon numbers and 
        background values with the ranges used during training.
        If the inferred values fall outside of the green regions, or are concentrated 
        in a small subspace of it, it might make sense to adjust the simulation 
        parameters and retrain the the network.""")
    
        plt.figure(figsize=(14,4))
        
        plt.subplot(131)
        mu, sig = param.Simulation.intensity_mu_sig
        plt.axvspan(0, mu+sig*3, color='green', alpha=0.1)
        sns.distplot(emitter.phot.numpy())
        plt.xlabel('Inferred number of photons')
        plt.xlim(0)
        
        plt.subplot(132)
        plt.axvspan(*param.Simulation.bg_uniform, color='green', alpha=0.1)
        sns.distplot(emitter.bg.numpy())
        plt.xlabel('Inferred background values')
        
        plt.show()

    # utils from:
    # https://github.com/TuragaLab/DECODE/blob/4c0f38c33681d1c3005bc18bd99a11f46d30f1c1/decode/utils/emitter_io.py
    def convert_dict_torch_list(data: dict) -> dict:
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.tolist()
        return data
    
    def convert_dict_torch_numpy(data: dict) -> dict:
        """Convert all torch tensors in dict to numpy."""
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.numpy()
        return data
    
    def change_to_one_dim(data: dict) -> dict:
        """
        Change xyz tensors to be one-dimensional.
    
        Args:
            data: emitterset as dictionary
    
        """
        xyz = data.pop('xyz')
        xyz_cr = data.pop('xyz_cr')
        xyz_sig = data.pop('xyz_sig')
    
        data_one_dim = {'x': xyz[:, 0], 'y': xyz[:, 1], 'z': xyz[:, 2]}
        data_one_dim.update(data)
        data_one_dim.update({'x_cr': xyz_cr[:, 0], 'y_cr': xyz_cr[:, 1], 'z_cr': xyz_cr[:, 2]})
        data_one_dim.update({'x_sig': xyz_sig[:, 0], 'y_sig': xyz_sig[:, 1], 'z_sig': xyz_sig[:, 2]})
    
        return data_one_dim

    emd = emitter.data
    data = copy.deepcopy(emd)
    
    data = change_to_one_dim(convert_dict_torch_numpy(data))
    data = pd.DataFrame.from_dict(data)

    data.rename(columns={"frame_ix": "frame"}, inplace=True)

    data = data[data['prob'] == prob]

    return data, proj_sd

def preview_locs(locs_df, s_marker=1, alpha_marker=0.1, color_marker='Blues_r', size_x=10, size_y=4, res_dpi=160, font=8):
    
    plt.rcParams.update({'font.size': font})
    plt.figure(figsize=(size_x, size_y), dpi=res_dpi)
    plt.scatter(locs_df['y'], locs_df['x'], c=locs_df['frame'], s=s_marker, alpha=alpha_marker, cmap=color_marker)
    plt.axis('scaled')
    plt.axis('off')
    plt.show()






