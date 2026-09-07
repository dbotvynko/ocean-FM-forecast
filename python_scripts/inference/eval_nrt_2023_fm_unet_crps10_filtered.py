"""
Same as eval_nrt_2023_fm_unet_crps10.py, except the ground truth pulled
from the NRT gridded product is sla_filtered instead of sla_unfiltered.

sla_unfiltered retains internal-tide/high-frequency signal that a
mesoscale mapping model (trained on filtered-ish glorys12 reanalysis SLA)
isn't trying to reproduce, so scoring against it can inflate RMSE/CRPS
relative to what the model can actually resolve. sla_filtered is the
standard OSE/DUACS-style scoring target for this reason.

Writes to a separate output directory from the unfiltered run so neither
overwrites the other's per-day files.

Usage:
    python eval_nrt_2023_fm_unet_crps10_filtered.py
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
NRT_2023_VAR = "sla_filtered"
OUT_DIR = "/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/eval_nrt2023_fm_unet_crps10_filtered/"
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

rmses, crps_fair_means = evaluator.run_year_mean_std_crps(start_dates, out_dir=OUT_DIR)

print()
print("Per-window forecast_mean/forecast_std/crps/crps_fair/truth saved to:", OUT_DIR)
print()
for lt in LEADTIMES:
    rmse_mean = np.nanmean(rmses[lt])
    crps_mean = np.nanmean(crps_fair_means[lt])
    n = len(rmses[lt])
    print(f"leadtime {lt}: mean RMSE over {n} windows = {rmse_mean:.5f}  |  mean fair CRPS = {crps_mean:.5f}")
