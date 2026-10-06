"""
Put the CLS 2024 gridded SLA products (Nadir without SWOT nadir, and Nadir + SWOT KaRIn, those used by the UNet runs)
on the FM model grid (lat -80..89.75, lon -180..179.75, 0.25 deg): longitudes padded beyond +-180 are dropped and
latitudes north of 81.75 deg are filled with NaN (no data). Variable sla_filtered, unchanged otherwise.

Usage: python prepare_cls_inputs_2024.py   # -> inputs/cls_{nadir,nadir_swot}_2024_fm_grid.nc
"""
from pathlib import Path

import numpy as np
import xarray as xr

SOURCES = {"nadir": "/Odyssey/public/swot_traces/cls/gridded_obs_sla_0.25_nadirs_2024_without_swon.nc",
           "nadir_swot": "/Odyssey/public/swot_traces/cls/gridded_obs_sla_0.25_swot_2024_reformated.nc"}
OUT = Path(__file__).resolve().parents[2] / "inputs"
LAT, LON = np.arange(-80.0, 89.76, 0.25), np.arange(-180.0, 179.76, 0.25)

OUT.mkdir(exist_ok=True)
for name, path in SOURCES.items():
    da = xr.open_dataset(path)["sla_filtered"].sel(lon=LON).reindex(lat=LAT)
    da.astype("float32").to_dataset().to_netcdf(OUT / f"cls_{name}_2024_fm_grid.nc", encoding={"sla_filtered": {"zlib": True, "complevel": 4}})
    print(name, dict(da.sizes), float(da.isel(time=100).notnull().mean()))
