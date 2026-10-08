"""
Convert the 2024 OceanBench member files (eval_2024_oceanbench_fm_unet_members.py: forecast(sample, leadtime, lat, lon),
truth) into the mean/std/CRPS window files of eval_2024_oceanbench_fm_unet_sla_filtered.py, with the same estimators
(YearlyLeadtimeEvaluator.day_result_to_mean_std_crps_dataset), so that a members run feeds the existing OceanBench
pipeline (oceanbench_eval/postprocess_fm.sh) unchanged. CPU only.

TRUTH_TAG: take `truth` (and score CRPS / RMSE against it) from another members run, e.g. the NRT nadir run for runs fed
with CLS Nadir + SWOT, whose own truth would contain SWOT pixels.

Usage:
    python members_to_mean_std_crps_2024.py TAG [TRUTH_TAG]   # outputs/eval_2024_oceanbench_members_TAG -> ..._wednesdays_TAG
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from contrib.generative.inference import crps_ensemble, crps_ensemble_fair  # noqa: E402

OUTPUTS = REPO_ROOT / "outputs"
tag = sys.argv[1]
truth_dir = OUTPUTS / f"eval_2024_oceanbench_members_{sys.argv[2] if len(sys.argv) > 2 else tag}"
in_dir, out_dir = OUTPUTS / f"eval_2024_oceanbench_members_{tag}", OUTPUTS / f"eval_2024_oceanbench_wednesdays_{tag}"
out_dir.mkdir(parents=True, exist_ok=True)

for path in sorted(in_dir.glob("2024-*.nc")):
    if (out_dir / path.name).exists():
        continue
    with xr.open_dataset(path) as ds:
        members = ds.forecast.values  # (sample, leadtime, lat, lon)
        with xr.open_dataset(truth_dir / path.name) as reference:
            truth = reference.truth.values  # (leadtime, lat, lon)
        mean = members.mean(axis=0)
        rmse = np.array([np.sqrt(np.nanmean((mean[k] - truth[k])[np.isfinite(truth[k])] ** 2)) for k in range(truth.shape[0])])
        variables = {
            "forecast_mean": mean, "forecast_std": members.std(axis=0),
            "crps": np.stack([crps_ensemble(members[:, k], truth[k]) for k in range(truth.shape[0])]),
            "crps_fair": np.stack([crps_ensemble_fair(members[:, k], truth[k]) for k in range(truth.shape[0])]),
            "truth": truth,
        }
        out = xr.Dataset({k: (("leadtime", "lat", "lon"), v.astype(np.float32)) for k, v in variables.items()},
                         coords={c: ds[c] for c in ("leadtime", "valid_time", "lat", "lon", "init_time")}, attrs=ds.attrs)
        out["rmse"] = ("leadtime", rmse.astype(np.float32))
        out.attrs["num_samples"] = members.shape[0]
    tmp = out_dir / (path.name + ".tmp")  # write then rename, so an interrupted write is never taken for a finished file
    out.to_netcdf(tmp, encoding={v: {"zlib": True, "complevel": 4} for v in out.data_vars})
    tmp.replace(out_dir / path.name)
    print(path.name, "rmse", np.round(rmse, 4), flush=True)
