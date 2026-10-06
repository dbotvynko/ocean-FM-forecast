"""
OceanBench 2024 FM UNet forecasts with every ensemble member saved: same windows, input and model as
eval_2024_oceanbench_fm_unet_sla_filtered.py (48 Wednesday starts, window starting 14 days before, lead days 0-6,
10 members, 2024 NRT nadir sla_filtered input), but each window file holds the raw members
forecast(sample, leadtime, lat, lon) and the L3 truth, for per-member diagnostics (spectra, hybrid ensembles).

The starts can be split over several jobs (--part i --nparts n takes every n-th start from i). Existing window files
are skipped, and files are written to a temporary name then renamed, so a requeued job resumes where it stopped.

Usage:
    python eval_2024_oceanbench_fm_unet_members.py --xp XP --ckpt CKPT --tag TAG [--part I --nparts N] [--input PATH --var VAR]
"""

import argparse
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
    coord_builder_for,
    load_gen_flow_checkpoint,
    load_gridded_sla,
)

parser = argparse.ArgumentParser()
parser.add_argument("--xp", required=True)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--tag", required=True)
parser.add_argument("--part", type=int, default=0)
parser.add_argument("--nparts", type=int, default=1)
parser.add_argument("--input", default="/Odyssey/public/altimetry_traces/2024/NRT_REAL_no_swon/gridded/gridded_input.nc",
                    help="gridded SLA input (default: 2024 NRT nadir without SWOT nadir; see prepare_cls_inputs_2024.py for CLS)")
parser.add_argument("--var", default="sla_filtered")
args = parser.parse_args()

WEDNESDAYS = pd.date_range("2024-01-17", "2024-12-11", freq="7D")
START_DATES = (WEDNESDAYS - pd.Timedelta(days=14))[args.part::args.nparts]  # window start; lead day 0 is valid on the Wednesday
OUT_DIR = Path(f"/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/eval_2024_oceanbench_members_{args.tag}/")
LEADTIMES = range(7)
NUM_SAMPLES = 10

with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "config")):
    cfg = compose(config_name="main", overrides=[f"xp={args.xp}"])

model = hydra.utils.instantiate(cfg.model)
model = load_gen_flow_checkpoint(model, args.ckpt)
print("xp:", args.xp, "| ckpt:", args.ckpt, "| input:", args.input, args.var, "| part", args.part, "of", args.nparts, flush=True)

domain_train = hydra.utils.instantiate(cfg.domain.train)
sla_da = load_gridded_sla(args.input, var=args.var, lat_slice=domain_train["lat"], lon_slice=domain_train["lon"])
evaluator = YearlyLeadtimeEvaluator(
    model,
    sla_da,
    tuple(cfg.datamodule.norm_stats.train),
    patch_time=cfg.datamodule.xrds_kw.train.patch_dims.time,
    leadtimes=LEADTIMES,
    num_samples=NUM_SAMPLES,
    coord_builder=coord_builder_for(cfg),
)

OUT_DIR.mkdir(parents=True, exist_ok=True)
for start in START_DATES:
    out_path = OUT_DIR / f"{start:%Y-%m-%d}.nc"
    if out_path.exists():
        continue
    ds = evaluator.day_result_to_dataset(start, evaluator.run_day(start))
    ds["forecast"] = ds["forecast"].astype(np.float32)
    ds["truth"] = ds["truth"].astype(np.float32)
    tmp_path = out_path.with_suffix(".nc.tmp")
    ds.to_netcdf(tmp_path, encoding={v: {"zlib": True, "complevel": 4} for v in ("forecast", "truth")})
    tmp_path.replace(out_path)
    print(start.date(), "rmse", np.round(ds.rmse.values, 4), flush=True)
