"""
Combine the per-day .nc files produced by
eval_nrt_2023_fm_unet_crps10_filtered.py into one NetCDF file per lead
time (test_leadtime_14.nc ... test_leadtime_20.nc), each holding every
available day concatenated along a `time` dimension.

Pointed at the sla_filtered 10-member CRPS run's output dir
(eval_nrt2023_fm_unet_crps10_filtered/, Jan-Mar window starts), so the
combined files carry forecast_mean/forecast_std/crps/crps_fair/truth
scored against sla_filtered. See combine_nrt_2023_leadtimes.py for the
sla_unfiltered counterpart.

Safe to run while eval_nrt_2023_fm_unet_crps10_filtered.py is still
writing new daily files -- it only combines whatever per-day files
already exist in OUT_DIR. Re-run again once the run has finished for the
complete files.

Usage:
    python combine_nrt_2023_leadtimes_filtered.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from contrib.generative.inference import combine_leadtime_files  # noqa: E402

OUT_DIR = "/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/eval_nrt2023_fm_unet_crps10_filtered/"
LEADTIMES = range(7)

written = combine_leadtime_files(OUT_DIR, leadtimes=LEADTIMES)

for lt, path in written.items():
    print(f"leadtime {lt}: {path}")
