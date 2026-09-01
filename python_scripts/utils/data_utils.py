import numpy as np
import xarray as xr
from pathlib import Path
import pyinterp
import pyinterp.fill
import pyinterp.backends.xarray

gf_domain = dict(lon=slice(-66, -54), lat=slice(32, 44))
full_var_transpose = ("time", "lat", "lon")

def load_nc_data(path, variables, time=None):

    ds =  xr.open_dataset(path).rename({'latitude':'lat', 'longitude':'lon'})
    #print(ds.data_vars)
    ds = ds.squeeze("depth")
    #print(ds.data_vars)
    ds = ds.sel(gf_domain)


    if time is not None:
        ds = ds.sel({"time":time})
        var_transpose = full_var_transpose[1:]
    else:
        var_transpose = full_var_transpose

    ds = (
            ds[variables]
            .load()
        )

    for var in variables:
        ds[var] = remove_nan(ds[var], var_transpose)

    return ds.transpose(*var_transpose).to_array()

def remove_nan(da, var_transpose):
    da['lon'] = da['lon'].assign_attrs(units="degrees_east")
    da['lat'] = da['lat'].assign_attrs(units="degrees_north")

    da.transpose(*var_transpose[::-1])[:, :] = pyinterp.fill.gauss_seidel(
        pyinterp.backends.xarray.Grid3D(da) if len(var_transpose) == 3 else pyinterp.backends.xarray.Grid2D(da)
    )[1]
    return da

def info_nc_data(path):
    ds = xr.open_dataset(path)

    print(ds.data_vars)

def load_altimetry_data(obs_from_tgt=False, time=None):
    path = '/DATASET/NATL/natl_gf_w_5nadirs.nc'

    ds =  (
        xr.open_dataset(path)
    )

    ds = ds.sel(gf_domain).sel({"time":time}).squeeze("time")

    if obs_from_tgt:
        ds = ds.assign(input=ds.tgt.where(np.isfinite(ds.input), np.nan))
    
    return (
        ds[["nadir_obs","ssh"]]
        .load()
        .transpose("lat", "lon")
        .to_array()
    )

def load_nadir_obs(time_sel):
    path = '/DATASET/NATL/natl_gf_w_5nadirs.nc'

    ds =  (
        xr.open_dataset(path)
    )

    ds = ds.sel(gf_domain).sel(time_sel)
    
    return (
        ds[["nadir_obs"]]
        .load()
        .transpose("time","lat", "lon")
        #.squeeze('variable')
        .to_array()
    )