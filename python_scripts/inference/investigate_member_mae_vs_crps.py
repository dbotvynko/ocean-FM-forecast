"""
Per-member MAE vs. CRPS-fair, leadtime 0 only, FM-UNet (forecast_DDPM_UNet_1patch).

None of the saved eval_nrt2023_fm_unet* runs kept the raw 10 ensemble
members (by design, to save disk -- see their own docstrings), so this
can't be computed from already-saved files. This re-runs actual model
inference, but only for a small evenly-spaced subset of the original
90-day Jan-Mar 2023 window (every 6th day -> 15 days) and leadtime 0 only,
to keep compute cheap (~15/90 of the full crps10 run's GPU cost) while
still giving a representative picture of the member-MAE-vs-CRPS
relationship.

For each sampled day, calls YearlyLeadtimeEvaluator.run_day() directly
(instead of run_year_mean_std_crps, which discards the raw ensemble after
computing summary stats) and keeps the raw (10, lat, lon) member forecasts
just long enough to compute:
  - per-member scalar MAE (spatial nanmean of |member - truth|), one per
    ensemble member
  - the ensemble-mean forecast's MAE
  - CRPS-fair (spatial nanmean), same estimator as the full crps10 run

Saves one row per sampled day to a small CSV (no raw fields), safe to
analyze without a GPU afterward.

Usage:
    python investigate_member_mae_vs_crps.py
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
    crps_ensemble_fair,
    load_gen_flow_checkpoint,
    load_gridded_sla,
)

CKPT_PATH = (
    "/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/2026-09-01/13-42-20/"
    "forecast_DDPM_UNet_1patch/checkpoints/val_loss=0.01161-epoch=153.ckpt"
)
NRT_2023_PATH = "/Odyssey/public/altimetry_traces/nrt_sla/2023/gridded_input.nc"
NRT_2023_VAR = "sla_unfiltered"
OUT_CSV = "/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/member_mae_vs_crps_leadtime0.csv"
LEADTIME = 0
NUM_SAMPLES = 10  # match the crps10 run's ensemble size for a fair comparison

with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "config")):
    cfg = compose(config_name="main", overrides=["xp=forecast_DDPM_UNet_1patch"])

model = hydra.utils.instantiate(cfg.model)
model = load_gen_flow_checkpoint(model, CKPT_PATH)

norm_stats = tuple(cfg.datamodule.norm_stats.train)
patch_time = cfg.datamodule.xrds_kw.train.patch_dims.time

domain_train = hydra.utils.instantiate(cfg.domain.train)
sla_da = load_gridded_sla(
    NRT_2023_PATH,
    var=NRT_2023_VAR,
    lat_slice=domain_train["lat"],
    lon_slice=domain_train["lon"],
)

all_start_dates = pd.date_range("2023-01-01", "2023-03-31", freq="D")
start_dates = all_start_dates[::6]  # 15 evenly-spaced days out of the 90
print(f"Sampling {len(start_dates)} of {len(all_start_dates)} days: {[str(d.date()) for d in start_dates]}")

evaluator = YearlyLeadtimeEvaluator(
    model, sla_da, norm_stats, patch_time=patch_time, leadtimes=[LEADTIME], num_samples=NUM_SAMPLES
)

rows = []
for start_date in start_dates:
    day_result = evaluator.run_day(start_date)
    pred = day_result[LEADTIME]["pred"]  # (num_samples, lat, lon)
    true = day_result[LEADTIME]["true"]  # (lat, lon)
    finite = np.isfinite(true)

    member_maes = [
        float(np.nanmean(np.abs(pred[m][finite] - true[finite]))) if finite.any() else np.nan
        for m in range(NUM_SAMPLES)
    ]
    ensemble_mean_mae = float(np.nanmean(np.abs(pred.mean(axis=0)[finite] - true[finite]))) if finite.any() else np.nan
    crps_fair_mean = float(np.nanmean(crps_ensemble_fair(pred, true)))
    rmse = day_result[LEADTIME]["rmse"]

    row = {
        "start_date": str(pd.Timestamp(start_date).date()),
        "ensemble_mean_mae": ensemble_mean_mae,
        "crps_fair_mean": crps_fair_mean,
        "rmse": rmse,
    }
    for m, mae in enumerate(member_maes):
        row[f"member_{m}_mae"] = mae
    rows.append(row)
    print(
        f"{row['start_date']}: ensemble_mean_mae={ensemble_mean_mae:.5f}  "
        f"crps_fair={crps_fair_mean:.5f}  member_mae_range=[{min(member_maes):.5f}, {max(member_maes):.5f}]"
    )

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")
