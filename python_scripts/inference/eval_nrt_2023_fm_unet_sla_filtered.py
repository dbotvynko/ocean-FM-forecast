"""
Copy of eval_nrt_2023_fm_unet_crps10.py (left untouched) for a filtered-input
run on the first 90 days of 2023:

  - input: 2023 NRT gridded nadir SLA, sla_filtered (instead of sla_unfiltered)
  - window starts 2023-01-01 .. 2023-03-31 (90 windows), lead times 0-6,
    10-member ensemble (same evaluator, so per-window files also carry
    crps/crps_fair/forecast_mean/forecast_std/truth)
  - RMSE (and mean bias) of the ensemble mean per lead time vs the reference
    SLA = the same L3 sla_filtered product, at its observed pixels on each
    forecast day (the forecast days' obs are never seen as input), written to
    <out_dir>/rmse_summary.csv

Usage:
    python eval_nrt_2023_fm_unet_sla_filtered.py --xp XP --ckpt CKPT --tag TAG
"""

import argparse
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import xarray as xr
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from contrib.generative.inference import (  # noqa: E402
    YearlyLeadtimeEvaluator,
    coord_builder_for,
    load_gen_flow_checkpoint,
    load_gridded_sla,
)

parser = argparse.ArgumentParser()
parser.add_argument("--xp", required=True)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--tag", required=True)
args = parser.parse_args()

NRT_2023_PATH = "/Odyssey/public/altimetry_traces/nrt_sla/2023/gridded_input.nc"
NRT_2023_VAR = "sla_filtered"
START_DATES = pd.date_range("2023-01-01", periods=90, freq="D")
OUT_DIR = Path(f"/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/eval_nrt2023_sla_filtered_90d_{args.tag}/")
LEADTIMES = range(7)
NUM_SAMPLES = 10

with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "config")):
    cfg = compose(config_name="main", overrides=[f"xp={args.xp}"])

model = hydra.utils.instantiate(cfg.model)
model = load_gen_flow_checkpoint(model, args.ckpt)
print("xp:", args.xp, "| ckpt:", args.ckpt, flush=True)

norm_stats = tuple(cfg.datamodule.norm_stats.train)
patch_time = cfg.datamodule.xrds_kw.train.patch_dims.time
domain_train = hydra.utils.instantiate(cfg.domain.train)
sla_da = load_gridded_sla(NRT_2023_PATH, var=NRT_2023_VAR,
                          lat_slice=domain_train["lat"], lon_slice=domain_train["lon"])

evaluator = YearlyLeadtimeEvaluator(
    model,
    sla_da,
    norm_stats,
    patch_time=patch_time,
    leadtimes=LEADTIMES,
    num_samples=NUM_SAMPLES,
    coord_builder=coord_builder_for(cfg),
)
rmses_obs, _ = evaluator.run_year_mean_std_crps(START_DATES, out_dir=str(OUT_DIR))

# RMSE / bias of the ensemble mean vs the L3 sla_filtered reference
# (`truth` in the per-window files), per lead time
rows = []
for start in START_DATES:
    ds = xr.open_dataset(OUT_DIR / f"{start:%Y-%m-%d}.nc")
    for lt in LEADTIMES:
        day = ds.isel(leadtime=lt)
        pred = day.forecast_mean.values
        ref = day.truth.values
        ok = np.isfinite(pred) & np.isfinite(ref)
        err = pred[ok] - ref[ok]
        rows.append(dict(start=start, leadtime=lt, n_obs=int(ok.sum()),
                         rmse=np.sqrt(np.mean(err ** 2)) if ok.any() else np.nan,
                         bias=err.mean() if ok.any() else np.nan))

per_window = pd.DataFrame(rows)
per_window.to_csv(OUT_DIR / "rmse_per_window.csv", index=False)
summary = per_window.groupby("leadtime")[["rmse", "bias", "n_obs"]].mean()
summary.to_csv(OUT_DIR / "rmse_summary.csv")

print()
print(f"{args.tag}: ensemble-mean RMSE / bias vs L3 sla_filtered over {len(START_DATES)} windows (m)")
print(summary.to_string(float_format=lambda x: f"{x:+.5f}"))
