"""
Compute geostrophic velocities (ugos, vgos, plus SLA-only anomalies
ugosa, vgosa) for test_leadtime_14.nc, using only sample=0 and only the
first 4 months of the year (to keep the output size manageable).

The physics (SLA-gradient anomaly + MDT-derived mean velocity) is
reproduced from CIA-Oceanix/global_ssh_forecasting_ose's
src/mod_velocities_geos.py:compute_geostrophic_velocity -- copied rather
than imported so this script has no dependency on that repo being cloned
or on PYTHONPATH. The only real dependency is the MDT NetCDF file itself
(cluster data, not code), at the same absolute path that repo's
retreive_geos_velocities() already reads from.

Like that repo's version, this interpolates our (coarser, ~0.25deg)
forecast/truth fields onto the MDT's native ~0.125deg (1/8deg) grid --
restricted to our own domain's lat/lon extent rather than the whole
globe, so it doesn't balloon the output for no reason (1/8deg vs 1/4deg
is already ~4x the pixels for the same area).

Usage:
    python compute_geos_velocities_leadtime14.py
"""

import numpy as np
import xarray as xr

IN_PATH = "/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/eval_nrt2023_fm_unet/test_leadtime_14.nc"
OUT_PATH = (
    "/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/eval_nrt2023_fm_unet/"
    "test_leadtime_14_sample0_geos_vel_Jan-Apr_MDTgrid.nc"
)
MDT_PATH = (
    "/Odyssey/public/duacs/cnes_obs-sl_glo_phy-mdt_my_0.125deg_P20Y_multi-vars_"
    "179.94W-179.94E_89.94S-89.94N_2003-01-01.nc"
)
VAR_NAME = "forecast"  # variable in the leadtime file to treat as SLA


def compute_geostrophic_velocity(lat, lon, sla, mdt_u, mdt_v):
    """
    lat, lon: 1D arrays (degrees). sla, mdt_u, mdt_v: (time, lat, lon)
    arrays. Returns (ug, vg, ug_anomaly, vg_anomaly), each (time, lat, lon).
    """
    g = 9.81  # gravity, m/s^2
    omega = 7.2921e-5  # Earth's rotation rate, rad/s

    f = 2 * omega * np.sin(np.radians(lat))
    f_masked = np.where(np.abs(lat) < 2, np.nan, f)  # equatorial singularity

    dlat = np.gradient(lat) * 111e3  # degrees -> meters
    dlon = np.gradient(lon) * 111e3
    dy = dlat
    dx = np.cos(np.radians(lat[:, None])) * dlon

    dSLA_dy = np.gradient(sla, axis=1) / dy[:, None]
    dSLA_dx = np.gradient(sla, axis=2) / dx

    vg_anomaly = g / f_masked[:, None] * dSLA_dx
    ug_anomaly = -g / f_masked[:, None] * dSLA_dy

    ug = ug_anomaly + mdt_u
    vg = vg_anomaly + mdt_v
    return ug, vg, ug_anomaly, vg_anomaly


ds = xr.open_dataset(IN_PATH).isel(sample=0, drop=True)
ds = ds.sel(time=slice("2023-01-01", "2023-04-30"))  # first 4 months only

mdt = xr.open_dataset(MDT_PATH).isel(time=0)
# MDT longitude is 0-360 native; wrap to -180..180, then sort both coords
# so .sel()/.interp() below can do straightforward ascending lookups.
mdt = mdt.assign_coords(longitude=(((mdt.longitude + 180) % 360) - 180))
mdt = mdt.sortby(["latitude", "longitude"])

# Restrict the MDT's native 1/8deg grid to our own domain extent (not the
# whole globe) before using it as the interpolation target.
mdt_domain = mdt.sel(
    latitude=slice(float(ds.lat.min()), float(ds.lat.max())),
    longitude=slice(float(ds.lon.min()), float(ds.lon.max())),
)
target_lat = mdt_domain.latitude.values.astype(np.float32)
target_lon = mdt_domain.longitude.values.astype(np.float32)

# Interpolate our (coarser) forecast/truth/rmse fields onto that grid.
ds = ds.interp(lat=target_lat, lon=target_lon)

# lat/lon are float64 in the source file; without this, mixing them into
# arithmetic with the float32 sla field silently upcasts every derived
# array (ugos, vgos, ugosa, vgosa, mdt_u, mdt_v) to float64, roughly
# doubling their footprint.
lat = ds["lat"].values.astype(np.float32)
lon = ds["lon"].values.astype(np.float32)
sla = ds[VAR_NAME].values.astype(np.float32)  # (time, lat, lon)
n_time = sla.shape[0]

# mdt_domain is already on exactly (target_lat, target_lon), so no further
# interpolation is needed for mdt_u/mdt_v.
mdt_u = np.repeat(mdt_domain["u"].values[np.newaxis, :, :], n_time, axis=0).astype(np.float32)
mdt_v = np.repeat(mdt_domain["v"].values[np.newaxis, :, :], n_time, axis=0).astype(np.float32)

ugos, vgos, ugosa, vgosa = compute_geostrophic_velocity(lat, lon, sla, mdt_u, mdt_v)

out = ds.copy()
out["ugos"] = (("time", "lat", "lon"), ugos.astype(np.float32))
out["vgos"] = (("time", "lat", "lon"), vgos.astype(np.float32))
out["ugosa"] = (("time", "lat", "lon"), ugosa.astype(np.float32))
out["vgosa"] = (("time", "lat", "lon"), vgosa.astype(np.float32))
out["mdt_u"] = (("time", "lat", "lon"), mdt_u)
out["mdt_v"] = (("time", "lat", "lon"), mdt_v)

# zlib compression (netcdf4 backend) -- to_netcdf writes uncompressed by
# default, which was most of the remaining size once float32 was fixed.
encoding = {var: {"zlib": True, "complevel": 4} for var in out.data_vars}
out.to_netcdf(OUT_PATH, encoding=encoding)
print("Saved:", OUT_PATH)
