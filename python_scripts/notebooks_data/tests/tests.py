import numpy as np
import xarray as xr
from pathlib import Path
import pyinterp
import pyinterp.fill

def remove_nan(da):
    da["lon"] = da.lon.assign_attrs(units="degrees_east")
    da["lat"] = da.lat.assign_attrs(units="degrees_north")

    da.transpose("lon", "lat", "time")[:, :] = pyinterp.fill.gauss_seidel(
        pyinterp.backends.xarray.Grid3D(da)
    )[1]
    return da

def load_nc_data(path, time=None):

    ds =  xr.open_dataset(path).rename({'latitude':'lat', 'longitude':'lon'})
    
    if time is not None:
        ds = ds.sel({"time":time})

    return (
        ds["zos"]
        .transpose("time", "lat", "lon")
    )

def test_nc_data(path):
    full_ds = xr.open_dataset(path)

    domain = dict(longitude=[-66, -54], latitude=[32, 44])
    ds = full_ds.isel(domain)

    print(ds.data_vars)
    print(ds.sizes)
        

def main():
    path="/DATASET/GLORYS12/reanalysis/cmems_mod_glo_phy_my_0.083deg_P1D-m_multi-vars_180.00W-179.92E_80.00S-90.00N_0.49m_2020-01-01-2020-12-31.nc"

    ds = test_nc_data(path)

if __name__=='__main__':
    main()