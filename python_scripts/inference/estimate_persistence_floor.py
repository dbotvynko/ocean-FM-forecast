"""
Estimate a persistence-forecast baseline for forecast_DDPM_UNet_1patch's
leadtime task: reconstruct the last observed day from its sparse mask
(Gauss-Seidel gap fill) and forecast every lead time (0-6) as that same
unchanged field. This is the standard reference in forecast verification
("skill relative to persistence") and, unlike the same-day-only proxy in
estimate_val_mse_floor.py, uses real temporal information -- it's just
naive about how the field evolves after the last observation.

Prints per-leadtime RMSE (normalized units, same convention as
test_leadtime_<idx>.nc's rmse variable), so it can be compared directly
against the model's own per-leadtime scores from eval_nrt_2023_fm_unet.py.

Does not touch the training run or its checkpoints; only reads the same
tgt netCDF and obs mask pickle used by the training config.

Usage:
    python estimate_persistence_floor.py
"""

import pickle
import sys
from pathlib import Path

import hydra
import numpy as np
import xarray as xr
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from contrib.generative.inference import (  # noqa: E402
    estimate_persistence_forecast_floor,
    load_gridded_sla,
)

NUM_WINDOWS = 30
SEED = 0
LEADTIMES = range(7)

with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "config")):
    cfg = compose(config_name="main", overrides=["xp=forecast_DDPM_UNet_1patch"])

tgt_path = cfg.datamodule.input_da.tgt_path
tgt_var = cfg.datamodule.input_da.tgt_var
mask_path = cfg.datamodule.input_da.inp_path
norm_stats = tuple(cfg.datamodule.norm_stats.train)
patch_time = cfg.datamodule.xrds_kw.train.patch_dims.time

domain_train = hydra.utils.instantiate(cfg.domain.train)
tgt_da = load_gridded_sla(
    tgt_path, var=tgt_var, lat_slice=domain_train["lat"], lon_slice=domain_train["lon"]
)

with open(mask_path, "rb") as f:
    daily_masks = pickle.load(f)
mask_da = xr.DataArray(
    ~np.isnan(np.stack(daily_masks, axis=0)),
    dims=("time", "lat", "lon"),
    coords={"lat": tgt_da.lat.values, "lon": tgt_da.lon.values},
)

result = estimate_persistence_forecast_floor(
    tgt_da,
    mask_da,
    norm_stats,
    leadtimes=LEADTIMES,
    num_windows=NUM_WINDOWS,
    seed=SEED,
    patch_time=patch_time,
)

print()
print(f"Persistence-forecast baseline over {NUM_WINDOWS} random windows:")
for lt, stats in result.items():
    print(
        f"  leadtime {lt} (window idx {patch_time // 2 + lt}): "
        f"mse={stats['mean_sq_error']:.5f}  rmse={stats['rmse_normalized']:.5f}"
    )
