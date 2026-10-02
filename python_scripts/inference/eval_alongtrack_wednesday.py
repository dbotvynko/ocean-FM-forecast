"""
Along-track scoring of saved 2023 NRT forecasts, restricted to the windows
whose lead time 0 is a Wednesday (lead time 1 = Thursday, ... 6 = Tuesday).

For each such window and lead time, the ensemble-mean forecast (the
forecast_mean already saved per window by eval_nrt_2023_fm_unet_sla_filtered.py)
is bilinearly interpolated to the along-track L3 points (nrt_sla concatenated
input, sla_filtered, 1 Hz) whose timestamp lies within +-0.5 day of the lead
time's date (date at 00:00, i.e. [D-12h, D+12h) -- the same binning the
daily gridded input was built with). RMSE and bias are computed over those
points, then averaged over windows. Also:
  - crps_gauss: CRPS at the same along-track points under a Gaussian
    approximation N(forecast_mean, forecast_std) of the ensemble (the raw
    members are not saved), both interpolated to the points;
  - crps_fair_grid: the exact fair ensemble CRPS already saved per pixel
    (10 members vs the daily gridded obs, i.e. the same +-0.5 day window),
    averaged over observed pixels.

No inference is run: it only reads saved forecasts, so it needs no GPU.

Usage:
    python eval_alongtrack_wednesday.py TAG [TAG ...]
reads outputs/eval_nrt2023_sla_filtered_90d_<TAG>/, writes
rmse_alongtrack_wednesday.csv there and prints a combined table.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from contrib.generative.inference import gaussian_crps  # noqa: E402

OUT = Path("/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs")
ALONGTRACK = "/Odyssey/public/altimetry_traces/nrt_sla/concat/concatenated_input.nc"
VAR = "sla_filtered"
HALF_WINDOW = np.timedelta64(12, "h")


def load_alongtrack(t0, t1):
    """Along-track points with t0 <= time < t1 (the file is sorted by time)."""
    ds = xr.open_dataset(ALONGTRACK)
    t = ds.time.values
    i0, i1 = np.searchsorted(t, t0), np.searchsorted(t, t1)
    pts = pd.DataFrame(dict(
        time=t[i0:i1],
        lat=ds.lat.values[i0:i1],
        lon=((ds.lon.values[i0:i1] + 180) % 360) - 180,
        obs=ds[VAR].values[i0:i1],
    ))
    return pts.dropna()


def bilinear(field, lat_grid, lon_grid, lat, lon):
    """Bilinear interpolation on a regular grid, periodic in lon; NaN if any corner is NaN."""
    dlat, dlon = lat_grid[1] - lat_grid[0], lon_grid[1] - lon_grid[0]
    fy = (lat - lat_grid[0]) / dlat
    fx = (lon - lon_grid[0]) / dlon
    y0, x0 = np.floor(fy).astype(int), np.floor(fx).astype(int)
    wy, wx = fy - y0, fx - x0
    inside = (y0 >= 0) & (y0 + 1 < len(lat_grid))
    y0c, y1c = np.clip(y0, 0, len(lat_grid) - 1), np.clip(y0 + 1, 0, len(lat_grid) - 1)
    x0c, x1c = x0 % len(lon_grid), (x0 + 1) % len(lon_grid)
    val = ((1 - wy) * (1 - wx) * field[y0c, x0c] + (1 - wy) * wx * field[y0c, x1c]
           + wy * (1 - wx) * field[y1c, x0c] + wy * wx * field[y1c, x1c])
    return np.where(inside, val, np.nan)


def score_day(rows, window, lt, valid_lt, day, pts, lat_grid, lon_grid):
    centre = np.datetime64(valid_lt.normalize())
    sel = pts[(pts.time >= centre - HALF_WINDOW) & (pts.time < centre + HALF_WINDOW)]
    pred = bilinear(day.forecast_mean.values, lat_grid, lon_grid, sel.lat.values, sel.lon.values)
    spread = bilinear(day.forecast_std.values, lat_grid, lon_grid, sel.lat.values, sel.lon.values)
    ok = np.isfinite(pred) & np.isfinite(spread)
    obs = sel.obs.values[ok]
    err = pred[ok] - obs
    crps_pts = gaussian_crps(pred[ok], np.maximum(spread[ok], 1e-6), obs)
    rows.append(dict(window=window, leadtime=int(lt), day=valid_lt.day_name(),
                     n_obs=int(ok.sum()),
                     rmse=np.sqrt(np.mean(err ** 2)), bias=err.mean(),
                     crps_gauss=crps_pts.mean(),
                     crps_fair_grid=float(day.crps_fair.mean(skipna=True))))


def score_tag(tag, pts):
    """
    Reads the per-window files (YYYY-MM-DD.nc) if present, otherwise the
    combined per-lead-time files (test_leadtime_<14+lt>.nc, one `time` entry
    per window at that lead time's valid date).
    """
    eval_dir = OUT / f"eval_nrt2023_sla_filtered_90d_{tag}"
    rows = []
    day_paths = sorted(eval_dir.glob("????-??-??.nc"))
    if day_paths:
        for path in day_paths:
            with xr.open_dataset(path) as ds:
                valid = pd.to_datetime(ds.valid_time.values)
                if valid[0].dayofweek != 2:  # lead time 0 must be a Wednesday
                    continue
                for lt in ds.leadtime.values:
                    score_day(rows, path.stem, lt, valid[lt], ds.sel(leadtime=lt), pts,
                              ds.lat.values, ds.lon.values)
    else:
        for lt in range(7):
            with xr.open_dataset(eval_dir / f"test_leadtime_{14 + lt}.nc") as ds:
                for t in pd.to_datetime(ds.time.values):
                    lt0 = t - pd.Timedelta(days=lt)
                    if lt0.dayofweek != 2:
                        continue
                    window = (lt0 - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
                    score_day(rows, window, lt, t, ds.sel(time=t), pts,
                              ds.lat.values, ds.lon.values)

    per_window = pd.DataFrame(rows)
    summary = per_window.groupby(["leadtime", "day"], sort=False).agg(
        rmse=("rmse", "mean"), bias=("bias", "mean"),
        crps_gauss=("crps_gauss", "mean"), crps_fair_grid=("crps_fair_grid", "mean"),
        n_obs=("n_obs", "mean"),
        n_windows=("window", "nunique"))
    summary.to_csv(eval_dir / "rmse_alongtrack_wednesday.csv")
    return summary


tags = sys.argv[1:]
pts = load_alongtrack(np.datetime64("2023-01-01T00:00"), np.datetime64("2023-05-01T00:00"))
print(f"{len(pts):,} along-track {VAR} points loaded (Jan-Apr 2023)", flush=True)

summaries = {tag: score_tag(tag, pts) for tag in tags}
for tag, s in summaries.items():
    print(f"\n{tag}  ({int(s.n_windows.iloc[0])} Wednesday windows)")
    print(s.to_string(float_format=lambda x: f"{x:.4f}"))
