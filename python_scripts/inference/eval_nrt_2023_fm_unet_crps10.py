"""
Evaluate the FM-UNet (forecast_DDPM_UNet_1patch) model on the 2023 NRT
gridded SLA product, per lead time (0-6 days), computing the exact
empirical CRPS (bias-corrected "fair" estimator, see
contrib.generative.inference.crps_ensemble_fair) from a 10-member
ensemble per window.

Unlike the Gaussian-approximation CRPS (compute_crps_gaussian.py, which
only needs the already-saved forecast_mean/forecast_std), this needs the
raw ensemble members while they're still in memory, so it's a fresh run
via YearlyLeadtimeEvaluator.run_year_mean_std_crps -- saves crps/
crps_fair/forecast_mean/forecast_std/truth/rmse per leadtime/pixel, not
every raw member (10 raw samples/day would multiply the on-disk size by
10 for no benefit once you only care about the ensemble summary).

Compute cost: ~2x the 5-member ensemble run (10 samples/window instead
of 5), same 90 Jan-Mar window starts -- budget for several hours.

Meant to run alongside an ongoing training job for the same xp: it only
reads the checkpoint file and never writes into the training run's output
directory, so it's safe to launch as a separate srun job in parallel.

Usage:
    python eval_nrt_2023_fm_unet_crps10.py
(edit CKPT_PATH below to point at whichever checkpoint you want to evaluate)
"""

import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from contrib.generative.inference import (  # noqa: E402
    YearlyLeadtimeEvaluator,
    load_gen_flow_checkpoint,
    load_gridded_sla,
)

CKPT_PATH = (
    "/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/2026-09-01/13-42-20/"
    "forecast_DDPM_UNet_1patch/checkpoints/val_loss=0.01161-epoch=153.ckpt"
)
NRT_2023_PATH = "/Odyssey/public/altimetry_traces/nrt_sla/2023/gridded_input.nc"
NRT_2023_VAR = "sla_unfiltered"
# Separate dir from the mean/std-only ensemble5 run: this one's daily
# files carry crps/crps_fair too, and a different num_samples.
OUT_DIR = "/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/eval_nrt2023_fm_unet_crps10/"
LEADTIMES = range(7)
NUM_SAMPLES = 10

with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "config")):
    cfg = compose(config_name="main", overrides=["xp=forecast_DDPM_UNet_1patch"])

model = hydra.utils.instantiate(cfg.model)
model = load_gen_flow_checkpoint(model, CKPT_PATH)

norm_stats = tuple(cfg.datamodule.norm_stats.train)
patch_time = cfg.datamodule.xrds_kw.train.patch_dims.time

# The raw NRT product is on the full global grid (lat=720); crop it to the
# same domain.train window (lat=680) the model was trained on.
domain_train = hydra.utils.instantiate(cfg.domain.train)
sla_da = load_gridded_sla(
    NRT_2023_PATH,
    var=NRT_2023_VAR,
    lat_slice=domain_train["lat"],
    lon_slice=domain_train["lon"],
)

start_dates = pd.date_range("2023-01-01", "2023-03-31", freq="D")  # Jan-Mar window starts only

evaluator = YearlyLeadtimeEvaluator(
    model,
    sla_da,
    norm_stats,
    patch_time=patch_time,
    leadtimes=LEADTIMES,
    num_samples=NUM_SAMPLES,
)

rmses = evaluator.run_year_mean_std_crps(start_dates, out_dir=OUT_DIR)

print()
print("Per-window forecast_mean/forecast_std/crps/crps_fair/truth saved to:", OUT_DIR)
print()
for lt, values in rmses.items():
    print(f"leadtime {lt}: mean RMSE over {len(values)} windows = {np.nanmean(values):.5f}")
