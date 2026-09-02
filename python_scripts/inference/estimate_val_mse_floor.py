"""
Estimate an empirical proxy for the irreducible val_loss floor of
forecast_DDPM_UNet_1patch: how much of the true SLA field's variance
simply cannot be recovered from the sparse 6-nadir observation network,
regardless of model quality.

This only estimates the t~0 end of the floor (see the long explanation in
contrib/generative/inference.py's estimate_obs_conditioned_residual_variance):
near t=0 in the flow-matching schedule, xt reveals the noise draw x0 but
nothing about tgt beyond what the observations constrain, so that's where
the achievable loss is *highest*. Near t=max_steps, xt~tgt almost exactly
and the achievable loss is close to 0. Uniformly averaging over t (as
training/validation do) would land the true floor somewhere between this
number and ~0 -- this script gives you the pessimistic end, which is still
a useful sanity check against the actual val_loss you're seeing.

Does NOT touch the training run or its checkpoints; only reads the same
tgt netCDF and obs mask pickle used by the training config.

Usage:
    python estimate_val_mse_floor.py
"""

import sys
from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from contrib.generative.inference import (  # noqa: E402
    estimate_obs_conditioned_residual_variance,
    load_gridded_sla,
)

NUM_DAYS = 30
SEED = 0

with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "config")):
    cfg = compose(config_name="main", overrides=["xp=forecast_DDPM_UNet_1patch"])

tgt_path = cfg.datamodule.input_da.tgt_path
tgt_var = cfg.datamodule.input_da.tgt_var
mask_path = cfg.datamodule.input_da.inp_path
norm_stats = tuple(cfg.datamodule.norm_stats.train)

# Defensive crop to the trained domain -- a no-op if tgt_path is already
# exactly that grid, a fix if it's the larger raw/global grid.
domain_train = hydra.utils.instantiate(cfg.domain.train)
tgt_da = load_gridded_sla(
    tgt_path, var=tgt_var, lat_slice=domain_train["lat"], lon_slice=domain_train["lon"]
)

import pickle  # noqa: E402

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

with open(mask_path, "rb") as f:
    daily_masks = pickle.load(f)
mask_da = xr.DataArray(
    ~np.isnan(np.stack(daily_masks, axis=0)),
    dims=("time", "lat", "lon"),
    coords={"lat": tgt_da.lat.values, "lon": tgt_da.lon.values},
)

result = estimate_obs_conditioned_residual_variance(
    tgt_da,
    mask_da,
    norm_stats,
    num_days=NUM_DAYS,
    seed=SEED,
    utils_dir=REPO_ROOT / "python_scripts" / "utils",
)

print()
print(f"Sampled {result['num_days']} random days, Gauss-Seidel baseline reconstruction:")
print(f"  mean squared error (normalized units): {result['mean_sq_error']:.5f}")
print(f"  rmse (normalized units):               {result['rmse_normalized']:.5f}")
print()
print("This is a pessimistic (t~0) proxy for the val_loss floor -- your")
print("current best val_loss can be compared against it, but the true")
print("floor (averaged over all diffusion steps) is somewhat below this.")
