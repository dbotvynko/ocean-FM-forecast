"""
Combine the per-day .nc files produced by eval_nrt_2023_fm_unet.py into one
NetCDF file per lead time (leadtime_0.nc ... leadtime_6.nc), each holding
every available day of 2023 concatenated along a `time` dimension.

Safe to run while eval_nrt_2023_fm_unet.py is still writing new daily
files -- it only combines whatever per-day files already exist in OUT_DIR.
Re-run again once the full year has finished for the complete files.

Usage:
    python combine_nrt_2023_leadtimes.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from contrib.generative.inference import combine_leadtime_files  # noqa: E402

OUT_DIR = "/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/eval_nrt2023_fm_unet/"
LEADTIMES = range(7)

written = combine_leadtime_files(OUT_DIR, leadtimes=LEADTIMES)

for lt, path in written.items():
    print(f"leadtime {lt}: {path}")
