import numpy as np
import pandas as pd
# from ..try_numba import try_numba_jit

import warnings
import logging

# from ..utils import (validate_tuple, guess_pos_columns, default_pos_columns,
#                      default_size_columns)
# from ..masks import (binary_mask, r_squared_mask,
#                      x_squared_masks, cosmask, sinmask)

# from ..try_numba import NUMBA_AVAILABLE, int, round

from trackpy.utils import validate_tuple, default_pos_columns, default_size_columns
from trackpy.masks import binary_mask, r_squared_mask, x_squared_masks, cosmask, sinmask
from trackpy.try_numba import NUMBA_AVAILABLE
from trackpy.refine.center_of_mass import (_refine, _numba_refine_2D, _numba_refine_2D_c,
                                    _numba_refine_2D_c_a, _numba_refine_3D)




def refine_com_mp(raw_image, image, radius, coords, max_iterations=10,
               engine='auto', shift_thresh=0.6, characterize=True,
               pos_columns=None):
    """Find the center of mass of a bright feature starting from an estimate.

    Characterize the neighborhood of a local maximum, and iteratively
    hone in on its center-of-brightness.

    Parameters
    ----------
    raw_image : array (any dimensions)
        Image used for final characterization. Ideally, pixel values of
        this image are not rescaled, but it can also be identical to
        ``image``.
    image : array (same size as raw_image)
        Processed image used for centroid-finding and most particle
        measurements.
    coords : array or DataFrame
        estimated position
    max_iterations : integer
        max number of loops to refine the center of mass, default 10
    engine : {'python', 'numba'}
        Numba is faster if available, but it cannot do walkthrough.
    shift_thresh : float, optional
        Default 0.6 (unit is pixels).
        If the brightness centroid is more than this far off the mask center,
        shift mask to neighboring pixel. The new mask will be used for any
        remaining iterations.
    characterize : boolean, True by default
        Compute and return mass, size, eccentricity, signal.
    pos_columns: list of strings, optional
        Column names that contain the position coordinates.
        Defaults to ``['y', 'x']`` or ``['z', 'y', 'x']``, if ``'z'`` exists.

    Returns
    -------
    DataFrame([x, y, mass, size, ecc, signal, raw_mass])
        where "x, y" are appropriate to the dimensionality of the image,
        mass means total integrated brightness of the blob,
        size means the radius of gyration of its Gaussian-like profile,
        ecc is its eccentricity (0 is circular),
        and raw_mass is the total integrated brightness in raw_image.
    """
    if isinstance(coords, pd.DataFrame):
        if pos_columns is None:
            pos_columns = guess_pos_columns(coords)
        index = coords.index
        coords = coords[pos_columns].values
    else:
        index = None

    radius = validate_tuple(radius, image.ndim)

    if pos_columns is None:
        pos_columns = default_pos_columns(image.ndim)
    columns = pos_columns + ['mass']
    if characterize:
        # isotropic = radius[1:] == radius[:-1]
        isotropic = False # we need Rg despite isotropic
        columns += default_size_columns(image.ndim, isotropic) + \
            ['ecc', 'signal', 'raw_mass']

    if len(coords) == 0:
        return pd.DataFrame(columns=columns)

    refined = refine_com_arr_mp(raw_image, image, radius, coords,
                             max_iterations=max_iterations,
                             engine=engine, shift_thresh=shift_thresh,
                             characterize=characterize)
    # print(f'refined shape{refined.shape}')
    # print(columns)
    return pd.DataFrame(refined, columns=columns, index=index)#['42']*refined.shape(1)





def refine_com_arr_mp(raw_image, image, radius, coords, max_iterations=10,
                   engine='auto', shift_thresh=0.6, characterize=True,
                   walkthrough=False):
    """Refine coordinates and return a numpy array instead of a DataFrame.

    See also
    --------
    refine_com
    """
    if max_iterations <= 0:
        warnings.warn("max_iterations has to be larger than 0. setting it to 1.")
        max_iterations = 1
    if raw_image.ndim != coords.shape[1]:
        raise ValueError("The image has a different number of dimensions than "
                         "the coordinate array.")

    # ensure that radius is tuple of integers, for direct calls to refine_com_arr()
    radius = validate_tuple(radius, image.ndim)
    # Main loop will be performed in separate function.
    if engine == 'auto':
        if NUMBA_AVAILABLE and image.ndim in [2, 3]:
            engine = 'numba'
        else:
            engine = 'python'

    # In here, coord is an integer. Make a copy, will not modify inplace.
    coords = np.round(coords).astype(int)

    if engine == 'python':
        results = _refine(raw_image, image, radius, coords, max_iterations,
                          shift_thresh, characterize, walkthrough)
    elif engine == 'numba':
        if not NUMBA_AVAILABLE:
            warnings.warn("numba could not be imported. Without it, the "
                          "'numba' engine runs very slow. Use the 'python' "
                          "engine or install numba.", UserWarning)
        if image.ndim not in [2, 3]:
            raise NotImplementedError("The numba engine only supports 2D or 3D "
                                      "images. You can extend it if you feel "
                                      "like a hero.")
        if walkthrough:
            raise ValueError("walkthrough is not availabe in the numba engine")
        # Do some extra prep in pure Python that can't be done in numba.
        N = coords.shape[0]
        mask = binary_mask(radius, image.ndim)
        if image.ndim == 3:
            if characterize:
                if np.all(radius[1:] == radius[:-1]):
                    results_columns = 8
                else:
                    results_columns = 10
            else:
                results_columns = 4
            r2_mask = r_squared_mask(radius, image.ndim)[mask]
            x2_masks = x_squared_masks(radius, image.ndim)
            z2_mask = image.ndim * x2_masks[0][mask]
            y2_mask = image.ndim * x2_masks[1][mask]
            x2_mask = image.ndim * x2_masks[2][mask]
            results = np.empty((N, results_columns), dtype=np.float64)
            maskZ, maskY, maskX = np.asarray(np.asarray(mask.nonzero()),
                                             dtype=np.int16)
            _numba_refine_3D(np.asarray(raw_image), np.asarray(image),
                             radius[0], radius[1], radius[2], coords, N,
                             int(max_iterations), shift_thresh,
                             characterize,
                             image.shape[0], image.shape[1], image.shape[2],
                             maskZ, maskY, maskX, maskX.shape[0],
                             r2_mask, z2_mask, y2_mask, x2_mask, results)
        elif not characterize:
            mask_coordsY, mask_coordsX = np.asarray(mask.nonzero(), np.int16)
            results = np.empty((N, 3), dtype=np.float64)
            _numba_refine_2D(np.asarray(image), radius[0], radius[1], coords, N,
                             int(max_iterations), shift_thresh,
                             image.shape[0], image.shape[1],
                             mask_coordsY, mask_coordsX, mask_coordsY.shape[0],
                             results)
        # elif radius[0] == radius[1]:
        #     mask_coordsY, mask_coordsX = np.asarray(mask.nonzero(), np.int16)
        #     results = np.empty((N, 7), dtype=np.float64)
        #     r2_mask = r_squared_mask(radius, image.ndim)[mask]
        #     cmask = cosmask(radius)[mask]
        #     smask = sinmask(radius)[mask]
        #     _numba_refine_2D_c(np.asarray(raw_image), np.asarray(image),
        #                        radius[0], radius[1], coords, N,
        #                        int(max_iterations), shift_thresh,
        #                        image.shape[0], image.shape[1], mask_coordsY,
        #                        mask_coordsX, mask_coordsY.shape[0],
        #                        r2_mask, cmask, smask, results)
        else:
            mask_coordsY, mask_coordsX = np.asarray(mask.nonzero(), np.int16)
            results = np.empty((N, 8), dtype=np.float64)
            x2_masks = x_squared_masks(radius, image.ndim)
            y2_mask = image.ndim * x2_masks[0][mask]
            x2_mask = image.ndim * x2_masks[1][mask]
            cmask = cosmask(radius)[mask]
            smask = sinmask(radius)[mask]
            _numba_refine_2D_c_a(np.asarray(raw_image), np.asarray(image),
                                 radius[0], radius[1], coords, N,
                                 int(max_iterations), shift_thresh,
                                 image.shape[0], image.shape[1], mask_coordsY,
                                 mask_coordsX, mask_coordsY.shape[0],
                                 y2_mask, x2_mask, cmask, smask, results)
    else:
        raise ValueError("Available engines are 'python' and 'numba'")

    return results