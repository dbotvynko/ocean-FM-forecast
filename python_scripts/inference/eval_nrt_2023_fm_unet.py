"""
Evaluate the FM-UNet (forecast_DDPM_UNet_1patch) model on the 2023 NRT
gridded SLA product, per lead time (0-6 days), using a trained checkpoint.

Runs a 5-member ensemble per window (each GenFlowLit.sample() call starts
from independent random noise) and saves only the ensemble mean and its
per-pixel std map -- not every raw member, which would multiply the
on-disk size by 5 for no benefit once you only care about the summary.
See YearlyLeadtimeEvaluator.run_year_mean_std / day_result_to_mean_std_dataset.

Meant to run alongside an ongoing training job for the same xp: it only
reads the checkpoint file and never writes into the training run's output
directory, so it's safe to launch as a separate srun job in parallel.

Only the datamodule.norm_stats and datamodule.xrds_kw.train.patch_dims.time
are read from the training config -- the (heavy) datamodule itself
(reanalysis + obs mask loading) is never instantiated here.

Usage:
    python eval_nrt_2023_fm_unet.py
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
# Separate dir from the completed single-sample run (eval_nrt2023_fm_unet/):
# run_year_mean_std writes a different schema (forecast_mean/forecast_std
# instead of forecast+sample dim), so reusing that dir would overwrite it.
OUT_DIR = "/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/eval_nrt2023_fm_unet_ensemble5/"
LEADTIMES = range(7)
NUM_SAMPLES = 5

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

rmses = evaluator.run_year_mean_std(start_dates, out_dir=OUT_DIR)

print()
print("Per-window forecasts/truth saved to:", OUT_DIR)
print()
for lt, values in rmses.items():
    print(f"leadtime {lt}: mean RMSE over {len(values)} windows = {np.nanmean(values):.5f}")
