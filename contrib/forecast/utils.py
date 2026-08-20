import numpy as np

from ocean4dvarnet.utils import get_constant_crop

def get_forecast_wei(patch_dims, **crop_kw):
    """
    return weight for forecast reconstruction:
    patch_dims: dimension of the patches used

    linear from 0 to 1 where there are obs
    linear from 1 to 0.5 for 7 days of forecast
    0 elsewhere
    """
    pw = get_constant_crop(patch_dims, **crop_kw)
    time_patch_weight = np.concatenate(
        (np.linspace(0, 1, (patch_dims['time'] - 1) // 2),
         np.linspace(1, 0.5, 7),
         np.zeros((patch_dims['time'] + 1) // 2 - 7)),
        axis=0)
    final_patch_weight = time_patch_weight[:, None, None] * pw
    return final_patch_weight