"""
Animated (MP4) film of per-member SLA evolution over the 90-day winter
(Jan-Mar 2023) window, Gulf Stream box only, leadtime 0, for the base FM
UNet (crps10) ensemble.

Unlike compute_member_spectra_fm_unet_crps10.py (which discards the raw
member fields immediately after computing their PSD, since keeping the
full grid for 90 days x 10 members would be large for no benefit), the
Gulf Stream box is tiny (10x10 deg at 1/4 deg res, ~40x40 pixels), so here
the raw per-member SLA fields themselves are the point and are kept and
saved to a small .npz, letting the animation be rebuilt without rerunning
the GPU inference.

Only 9 of the 10 raw members are plotted (member index 9 dropped), so
they fit a clean 3x3 grid -- one panel per member, all sharing a single
color scale (robust 1st/99th percentile over the whole run) so the panels
are visually comparable to each other and across days.

Same per-day GPU cost as compute_member_spectra_fm_unet_crps10.py /
eval_nrt_2023_fm_unet_crps10.py: one model.sample() call per day
regardless of how many leadtimes are requested, so leadtime=[0] here is
no cheaper than requesting more.

Frames are labelled with the forecast *valid* date (init date of the
14-day observation window + obs_days + leadtime), plus the init date.

Usage:
    python make_member_sla_animation_fm_unet_crps10.py [--leadtime 0] [--from-npz]

--from-npz skips inference and rebuilds the animation from the saved .npz.
"""

import argparse
import sys
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
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
from compute_spectra_deterministic_vs_ensemble_crps10 import LAT_BOUNDS, LON_BOUNDS  # noqa: E402

CKPT_PATH = (
    "/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/2026-09-01/13-42-20/"
    "forecast_DDPM_UNet_1patch/checkpoints/val_loss=0.01161-epoch=153.ckpt"
)
NRT_2023_PATH = "/Odyssey/public/altimetry_traces/nrt_sla/2023/gridded_input.nc"
NRT_2023_VAR = "sla_unfiltered"
NUM_SAMPLES = 10
NUM_MEMBERS_TO_PLOT = 9  # 3x3 grid; drops the 10th raw member
LEADTIME = 0  # overridden by --leadtime
OUT_DIR = Path("/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/figures/")
NPZ_DIR = Path("/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/member_animation_crps10/")


def run_all_days(evaluator, start_dates):
    lat = evaluator.sla_da.lat.values
    lon = evaluator.sla_da.lon.values
    latmask = (lat >= LAT_BOUNDS[0]) & (lat <= LAT_BOUNDS[1])
    lonmask = (lon >= LON_BOUNDS[0]) & (lon <= LON_BOUNDS[1])
    sub_lat = lat[latmask]
    sub_lon = lon[lonmask]

    fields = np.full((len(start_dates), NUM_SAMPLES, sub_lat.size, sub_lon.size), np.nan, dtype=np.float32)

    for i, start_date in enumerate(start_dates):
        day_result = evaluator.run_day(start_date)
        pred = day_result[LEADTIME]["pred"]  # (num_samples, lat, lon)
        pred_box = pred[:, latmask][:, :, lonmask]
        fields[i] = pred_box
        print(f"[{i + 1}/{len(start_dates)}] {pd.Timestamp(start_date).date()}: sampled")

    return fields, sub_lat, sub_lon


def build_animation(fields, sub_lat, sub_lon, dates, obs_days, out_path):
    members = fields[:, :NUM_MEMBERS_TO_PLOT]  # (n_days, 9, lat, lon)
    init_dates = pd.to_datetime([str(d) for d in dates])
    valid_dates = init_dates + pd.Timedelta(days=obs_days + LEADTIME)
    vmin, vmax = np.nanpercentile(members, [1, 99])
    vabs = max(abs(vmin), abs(vmax))

    nrows, ncols = 3, 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(11, 10), sharex=True, sharey=True)
    meshes = []
    for m, ax in enumerate(axs.ravel()):
        mesh = ax.pcolormesh(
            sub_lon, sub_lat, members[0, m], cmap="RdBu_r", vmin=-vabs, vmax=vabs, shading="auto",
        )
        ax.set_title(f"Member {m}", fontsize=10)
        meshes.append(mesh)
    fig.colorbar(meshes[0], ax=axs, orientation="vertical", shrink=0.8, label="SLA [m]")
    suptitle = fig.suptitle("", fontsize=13)

    def update(frame):
        for m, mesh in enumerate(meshes):
            mesh.set_array(members[frame, m].ravel())
        suptitle.set_text(
            f"FM UNet (crps10) ensemble members -- Gulf Stream SLA, leadtime {LEADTIME}d\n"
            f"valid {valid_dates[frame].date()} (init {init_dates[frame].date()}) ({frame + 1}/{len(dates)})"
        )
        return meshes + [suptitle]

    anim = animation.FuncAnimation(fig, update, frames=len(dates), blit=False)
    anim.save(out_path, writer="ffmpeg", fps=1, dpi=130)
    plt.close(fig)


def main(from_npz):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NPZ_DIR.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "config")):
        cfg = compose(config_name="main", overrides=["xp=forecast_DDPM_UNet_1patch"])

    patch_time = cfg.datamodule.xrds_kw.train.patch_dims.time
    obs_days = patch_time // 2
    npz_path = NPZ_DIR / f"member_fields_winter_leadtime{LEADTIME}.npz"
    out_path = OUT_DIR / f"member_sla_animation_fm_unet_crps10_winter_leadtime{LEADTIME}.mp4"

    if from_npz:
        d = np.load(npz_path)
        build_animation(d["fields"], d["lat"], d["lon"], d["dates"], obs_days, out_path)
        print("Saved animation:", out_path)
        return

    model = hydra.utils.instantiate(cfg.model)
    model = load_gen_flow_checkpoint(model, CKPT_PATH)

    norm_stats = tuple(cfg.datamodule.norm_stats.train)

    domain_train = hydra.utils.instantiate(cfg.domain.train)
    sla_da = load_gridded_sla(
        NRT_2023_PATH, var=NRT_2023_VAR, lat_slice=domain_train["lat"], lon_slice=domain_train["lon"],
    )

    start_dates = pd.date_range("2023-01-01", "2023-03-31", freq="D")  # winter, 90 days
    evaluator = YearlyLeadtimeEvaluator(
        model, sla_da, norm_stats, patch_time=patch_time, leadtimes=[LEADTIME], num_samples=NUM_SAMPLES,
    )

    print(f"Running {len(start_dates)}-day, {NUM_SAMPLES}-member inference, leadtime {LEADTIME}, Gulf Stream box...")
    fields, sub_lat, sub_lon = run_all_days(evaluator, start_dates)

    np.savez(
        npz_path, fields=fields, lat=sub_lat, lon=sub_lon,
        dates=np.array([str(d.date()) for d in start_dates]),
    )
    print("Saved raw member fields:", npz_path)

    build_animation(fields, sub_lat, sub_lon, start_dates, obs_days, out_path)
    print("Saved animation:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--leadtime", type=int, default=LEADTIME)
    parser.add_argument("--from-npz", action="store_true")
    args = parser.parse_args()
    LEADTIME = args.leadtime
    main(args.from_npz)
