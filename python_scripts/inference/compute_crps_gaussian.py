"""
Compute a Gaussian-approximation CRPS from the already-saved
forecast_mean/forecast_std files (eval_nrt2023_fm_unet_ensemble5/), with
no rerun of the model needed.

This is an approximation: it treats the ensemble's distribution at each
pixel/leadtime as Gaussian(forecast_mean, forecast_std) and uses the
closed-form CRPS formula (Gneiting & Raftery 2007) -- see
contrib.generative.inference.gaussian_crps. For the exact empirical CRPS
(no Gaussian assumption), the raw ensemble is needed, which means using
YearlyLeadtimeEvaluator.run_year_mean_std_crps on a new run instead (see
contrib/generative/inference.py) -- this script only works on data that's
already on disk.

Usage:
    python compute_crps_gaussian.py
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from contrib.generative.inference import gaussian_crps  # noqa: E402

IN_DIR = Path(
    "/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/eval_nrt2023_fm_unet_ensemble5/"
)
LEADTIME_INDICES = [14, 15, 16, 17, 18, 19, 20]  # test_leadtime_<idx>.nc, leadtime 0-6

for idx in LEADTIME_INDICES:
    path = IN_DIR / f"test_leadtime_{idx}.nc"
    if not path.exists():
        print(f"skip (not found): {path}")
        continue

    ds = xr.open_dataset(path)
    crps = gaussian_crps(ds["forecast_mean"].values, ds["forecast_std"].values, ds["truth"].values)

    ds_out = ds.assign(crps_gaussian=(("time", "lat", "lon"), crps.astype(np.float32)))
    encoding = {"crps_gaussian": {"zlib": True, "complevel": 4}}
    out_path = IN_DIR / f"test_leadtime_{idx}_crps.nc"
    ds_out.to_netcdf(out_path, encoding=encoding)

    mean_crps = float(np.nanmean(crps))
    print(f"leadtime idx {idx}: mean Gaussian CRPS = {mean_crps:.5f} -> {out_path}")
