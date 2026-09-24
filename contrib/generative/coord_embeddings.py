"""
Per-pixel lat/lon/day-of-year coordinate channels for the FM UNet, to test
whether conditioning the model on absolute position and season (rather than
only the masked SLA observations) improves the forecast.

Lat is not cyclic (bounded -90/90), so a single sin(lat) channel is enough.
Lon and day-of-year are both cyclic (the antimeridian, and the Dec31->Jan1
wraparound), so each gets a sin/cos pair. Day-of-year is computed per
timestep -- one sin/cos pair per stacked day in the patch -- rather than a
single value for the whole patch, so the model sees date drift across the
patch the same way it already does for the SLA channels.
"""

import numpy as np
import pandas as pd


def build_coord_channels(lat, lon, times):
    """
    lat: (nlat,) latitudes in degrees
    lon: (nlon,) longitudes in degrees
    times: (ntime,) array-like of datetime64 values, one per stacked day

    Returns a float32 array of shape (3 + 2 * ntime, nlat, nlon):
        [0]                 sin(lat)
        [1:3]               sin(lon), cos(lon)
        [3:3 + 2*ntime]     per-timestep sin(doy), cos(doy), interleaved
    """
    nlat, nlon = lat.shape[0], lon.shape[0]

    lat_rad = np.deg2rad(lat)[:, None]  # (nlat, 1)
    lon_rad = np.deg2rad(lon)[None, :]  # (1, nlon)

    lat_channel = np.broadcast_to(np.sin(lat_rad), (nlat, nlon))
    lon_sin = np.broadcast_to(np.sin(lon_rad), (nlat, nlon))
    lon_cos = np.broadcast_to(np.cos(lon_rad), (nlat, nlon))

    doy = pd.DatetimeIndex(np.asarray(times)).dayofyear.to_numpy()  # (ntime,)
    doy_frac = 2 * np.pi * doy / 365.25
    doy_sin = np.sin(doy_frac)
    doy_cos = np.cos(doy_frac)

    ntime = doy.shape[0]
    doy_channels = np.empty((2 * ntime, nlat, nlon), dtype=np.float32)
    doy_channels[0::2] = doy_sin[:, None, None]
    doy_channels[1::2] = doy_cos[:, None, None]

    coords = np.concatenate(
        [lat_channel[None], lon_sin[None], lon_cos[None], doy_channels], axis=0,
    )
    return coords.astype(np.float32)


def num_coord_channels(ntime):
    """Channel count returned by build_coord_channels for a patch of `ntime` days."""
    return 3 + 2 * ntime


# Fourier (multi-frequency) encoding
# ----------------------------------
# Harmonics are integers so lon and day-of-year stay exactly periodic.
# Day-of-year varies by < 30 days over a 29-day patch, so a single value
# (the patch's middle day) is used instead of one sin/cos pair per timestep:
# it keeps the channel count small (20 instead of 61) while the higher
# spatial harmonics give the UNet position information at finer scales
# than a single sin/cos can (sin(lat) alone barely changes near the poles).
LAT_HARMONICS = (1, 2, 4, 8)
LON_HARMONICS = (1, 2, 4, 8)
DOY_HARMONICS = (1, 2)


def build_fourier_coord_channels(lat, lon, times,
                                 lat_harmonics=LAT_HARMONICS,
                                 lon_harmonics=LON_HARMONICS,
                                 doy_harmonics=DOY_HARMONICS):
    """
    lat: (nlat,) latitudes in degrees
    lon: (nlon,) longitudes in degrees
    times: (ntime,) array-like of datetime64 values, one per stacked day

    Returns a float32 array of shape (num_fourier_coord_channels(), nlat, nlon):
        for k in lat_harmonics: sin(k*lat), cos(k*lat)
        for k in lon_harmonics: sin(k*lon), cos(k*lon)
        for k in doy_harmonics: sin(k*doy), cos(k*doy)   (patch middle day)
    """
    nlat, nlon = lat.shape[0], lon.shape[0]
    lat_rad = np.deg2rad(lat)[:, None]
    lon_rad = np.deg2rad(lon)[None, :]

    doy = pd.DatetimeIndex(np.asarray(times)).dayofyear.to_numpy()
    doy_frac = 2 * np.pi * doy[len(doy) // 2] / 365.25

    channels = []
    for k in lat_harmonics:
        channels += [np.sin(k * lat_rad), np.cos(k * lat_rad)]
    for k in lon_harmonics:
        channels += [np.sin(k * lon_rad), np.cos(k * lon_rad)]
    for k in doy_harmonics:
        channels += [np.full((1, 1), np.sin(k * doy_frac)), np.full((1, 1), np.cos(k * doy_frac))]

    coords = np.stack([np.broadcast_to(c, (nlat, nlon)) for c in channels], axis=0)
    return coords.astype(np.float32)


def num_fourier_coord_channels(lat_harmonics=LAT_HARMONICS,
                               lon_harmonics=LON_HARMONICS,
                               doy_harmonics=DOY_HARMONICS):
    """Channel count returned by build_fourier_coord_channels."""
    return 2 * (len(lat_harmonics) + len(lon_harmonics) + len(doy_harmonics))
