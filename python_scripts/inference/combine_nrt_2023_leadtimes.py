"""
Combine the per-day .nc files produced by eval_nrt_2023_fm_unet.py into one
NetCDF file per lead time (test_leadtime_14.nc ... test_leadtime_20.nc),
each holding every available day concatenated along a `time` dimension.

Currently pointed at the 5-member ensemble run's output dir
(eval_nrt2023_fm_unet_ensemble5/, Jan-Mar window starts), so the combined
files will carry forecast_mean/forecast_std instead of the single-sample
run's forecast(sample, ...). Point OUT_DIR back at eval_nrt2023_fm_unet/
for the original single-sample files instead.

Safe to run while eval_nrt_2023_fm_unet.py is still writing new daily
files -- it only combines whatever per-day files already exist in OUT_DIR.
Re-run again once the run has finished for the complete files.

Usage:
    python combine_nrt_2023_leadtimes.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from contrib.generative.inference import combine_leadtime_files  # noqa: E402

OUT_DIR = "/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/eval_nrt2023_fm_unet_ensemble5/"
LEADTIMES = range(7)

written = combine_leadtime_files(OUT_DIR, leadtimes=LEADTIMES)

for lt, path in written.items():
    print(f"leadtime {lt}: {path}")
