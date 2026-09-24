"""
Per-member wavenumber power spectra of the FM UNet (crps10) ensemble
forecast, compared against the deterministic UNet SLA reconstruction and
regridded GLO12 -- the per-member counterpart of
compute_spectra_deterministic_vs_ensemble_crps10.py, which could only show
the ensemble as mean +/- 1 std since raw members were never saved to disk.

This reruns actual model inference over all 90 start dates of a given
season (--season winter: Jan-Mar 2023; --season summer: Jul-Sep 2023)
with the full 10-member ensemble (leadtimes 0/3/5 by default -- same cost
as the original eval_nrt_2023_fm_unet_crps10.py run, since
YearlyLeadtimeEvaluator.run_day() draws its num_samples model.sample()
calls once per day and slices out every requested leadtime from that same
set of samples, so requesting fewer leadtimes here doesn't save GPU time).
The raw (10, lat, lon) members are cropped to the Gulf Stream box, each
member's isotropic PSD is computed immediately, and the raw fields are
then discarded -- never written to disk, since keeping the full-grid
ensemble for 90 days x 3 leadtimes would be large for no benefit once
only the small box's spectrum is needed.

Deterministic/GLO12 matching, the Gulf Stream box, and the isotropic PSD
routine are identical to (and imported from)
compute_spectra_deterministic_vs_ensemble_crps10.py, so all curves share
the same box, grid and PSD estimator as the mean/std version.

Saves the per-day per-member PSDs to a .npz per leadtime (cheap, and lets
the plot be redrawn/restyled without rerunning the GPU job) plus the
comparison figure itself.

Usage:
    python compute_member_spectra_fm_unet_crps10.py [--leadtimes 0 3 5] [--season winter|summer]
"""

import argparse
import sys
import warnings
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from hydra import compose, initialize_config_dir

warnings.filterwarnings("ignore", message="Isotropic wavenumber larger than the Nyquist wavenumber.*")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from contrib.generative.inference import (  # noqa: E402
    YearlyLeadtimeEvaluator,
    load_gen_flow_checkpoint,
    load_gridded_sla,
)
from compute_spectra_deterministic_vs_ensemble_crps10 import (  # noqa: E402
    DET_DIR,
    GLO12_CROP_PAD,
    GLO12_LAT,
    GLO12_LON,
    LAT_BOUNDS,
    LON_BOUNDS,
    build_glo12_date_index,
    isotropic_psd,
    regrid_to_target,
)

CKPT_PATH = (
    "/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/2026-09-01/13-42-20/"
    "forecast_DDPM_UNet_1patch/checkpoints/val_loss=0.01161-epoch=153.ckpt"
)
NRT_2023_PATH = "/Odyssey/public/altimetry_traces/nrt_sla/2023/gridded_input.nc"
NRT_2023_VAR = "sla_unfiltered"
NUM_SAMPLES = 10
OUT_DIR = Path("/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/figures/")
NPZ_DIR = Path("/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/member_spectra_crps10/")
SEASON_RANGES = {
    "winter": ("2023-01-01", "2023-03-31"),  # original 90-day Jan-Mar window
    "summer": ("2023-07-01", "2023-09-28"),  # 90-day Jul-Sep window
}


def run_all_leadtimes(evaluator, obs_days, start_dates, leadtimes):
    ds_dets = {lt: xr.open_dataset(DET_DIR / f"test_data_{14 + lt}.nc") for lt in leadtimes}
    glo12_indices = {lt: build_glo12_date_index(lt) for lt in leadtimes}

    lat = evaluator.sla_da.lat.values
    lon = evaluator.sla_da.lon.values
    latmask = (lat >= LAT_BOUNDS[0]) & (lat <= LAT_BOUNDS[1])
    lonmask = (lon >= LON_BOUNDS[0]) & (lon <= LON_BOUNDS[1])
    sub_lat = lat[latmask]
    sub_lon = lon[lonmask]

    glo12_latmask = (GLO12_LAT >= LAT_BOUNDS[0] - GLO12_CROP_PAD) & (GLO12_LAT <= LAT_BOUNDS[1] + GLO12_CROP_PAD)
    glo12_lonmask = (GLO12_LON >= LON_BOUNDS[0] - GLO12_CROP_PAD) & (GLO12_LON <= LON_BOUNDS[1] + GLO12_CROP_PAD)
    glo12_sub_lat = GLO12_LAT[glo12_latmask]
    glo12_sub_lon = GLO12_LON[glo12_lonmask]

    freq_r = None
    psd_members = {lt: [] for lt in leadtimes}
    psd_det = {lt: [] for lt in leadtimes}
    psd_glo12 = {lt: [] for lt in leadtimes}
    n_matched = {lt: 0 for lt in leadtimes}
    n_matched_glo12 = {lt: 0 for lt in leadtimes}

    for i, start_date in enumerate(start_dates):
        start_ts = pd.Timestamp(start_date)
        valid_times = {lt: start_ts + pd.Timedelta(days=int(obs_days + lt)) for lt in leadtimes}

        det_slices = {}
        for lt in leadtimes:
            try:
                det_slices[lt] = ds_dets[lt]["out"].sel(time=valid_times[lt]).values
            except KeyError:
                continue
        if not det_slices:
            print(f"[{i + 1}/{len(start_dates)}] {start_ts.date()}: no leadtime has a matching "
                  f"deterministic date, skipping (no inference run)")
            continue

        day_result = evaluator.run_day(start_date)  # one GPU pass, reused across all leadtimes below

        for lt, det_slice in det_slices.items():
            pred = day_result[lt]["pred"]  # (num_samples, lat, lon)
            det_box = det_slice[np.ix_(latmask, lonmask)]

            member_psds = []
            for m in range(pred.shape[0]):
                member_box = pred[m][np.ix_(latmask, lonmask)]
                iso = isotropic_psd(member_box, sub_lat, sub_lon)
                if freq_r is None:
                    freq_r = iso["freq_r"].values
                member_psds.append(iso.values)
            psd_members[lt].append(member_psds)
            psd_det[lt].append(isotropic_psd(det_box, sub_lat, sub_lon).values)
            n_matched[lt] += 1

            glo12_path = glo12_indices[lt].get(valid_times[lt].strftime("%Y-%m-%d"))
            if glo12_path is not None:
                with xr.open_dataset(glo12_path) as ds_glo12:
                    zos = ds_glo12["zos"].isel(time=0).values
                glo12_box = zos[np.ix_(glo12_latmask, glo12_lonmask)]
                glo12_box = regrid_to_target(glo12_box, glo12_sub_lat, glo12_sub_lon, sub_lat, sub_lon)
                psd_glo12[lt].append(isotropic_psd(glo12_box, sub_lat, sub_lon).values)
                n_matched_glo12[lt] += 1

        print(f"[{i + 1}/{len(start_dates)}] {start_ts.date()}: matched leadtimes {sorted(det_slices)}")

    for ds in ds_dets.values():
        ds.close()

    results = {}
    for lt in leadtimes:
        results[lt] = dict(
            freq_r=freq_r,
            psd_members=np.array(psd_members[lt]),  # (n_matched, num_samples, nfreq)
            psd_det=np.nanmean(psd_det[lt], axis=0),
            n_matched=n_matched[lt],
            psd_glo12=np.nanmean(psd_glo12[lt], axis=0) if psd_glo12[lt] else None,
            n_matched_glo12=n_matched_glo12[lt],
        )
    return results


def plot_member_spectra(leadtime, freq_r, psd_members, psd_det, n_matched, psd_glo12, n_matched_glo12, season_label):
    fig, ax = plt.subplots(figsize=(7, 6))

    num_samples = psd_members.shape[1]
    mean_over_days = np.nanmean(psd_members, axis=0)  # (num_samples, nfreq)
    colors = plt.cm.Blues(np.linspace(0.4, 0.95, num_samples))
    for m in range(num_samples):
        ax.plot(freq_r, mean_over_days[m], color=colors[m], linewidth=1.2, alpha=0.85,
                 label=f"Ensemble members ({num_samples})" if m == 0 else None)

    ax.plot(freq_r, psd_det, color="#eb6834", linewidth=2, linestyle="--", label="Deterministic UNet")
    if psd_glo12 is not None:
        ax.plot(freq_r, psd_glo12, color="#2a2a2a", linewidth=2, linestyle=":",
                 label=f"GLO12 (CMEMS forecast, {n_matched_glo12} weeks, regridded)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber [cycles/km]")
    ax.set_ylabel("PSD [m$^2$/(cycles/km)]")
    ax.set_title(
        f"Gulf Stream box wavenumber spectra per member -- leadtime {leadtime}d\n"
        f"FM UNet ensemble members vs. deterministic UNet vs. GLO12, 2023 {season_label} NRT mean over {n_matched} days"
    )
    ax.legend(frameon=False)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.3)
    fig.tight_layout()
    return fig


def main(leadtimes, season):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NPZ_DIR.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "config")):
        cfg = compose(config_name="main", overrides=["xp=forecast_DDPM_UNet_1patch"])

    model = hydra.utils.instantiate(cfg.model)
    model = load_gen_flow_checkpoint(model, CKPT_PATH)

    norm_stats = tuple(cfg.datamodule.norm_stats.train)
    patch_time = cfg.datamodule.xrds_kw.train.patch_dims.time
    obs_days = patch_time // 2

    domain_train = hydra.utils.instantiate(cfg.domain.train)
    sla_da = load_gridded_sla(
        NRT_2023_PATH, var=NRT_2023_VAR, lat_slice=domain_train["lat"], lon_slice=domain_train["lon"],
    )

    season_start, season_end = SEASON_RANGES[season]
    start_dates = pd.date_range(season_start, season_end, freq="D")  # full 90-day window
    evaluator = YearlyLeadtimeEvaluator(
        model, sla_da, norm_stats, patch_time=patch_time, leadtimes=leadtimes, num_samples=NUM_SAMPLES,
    )

    print(f"Running {len(start_dates)}-day ({season_start} to {season_end}), {NUM_SAMPLES}-member "
          f"inference for leadtimes {leadtimes}...")
    results = run_all_leadtimes(evaluator, obs_days, start_dates, leadtimes)

    for leadtime in leadtimes:
        r = results[leadtime]
        print(f"  leadtime={leadtime}: matched {r['n_matched']} days; "
              f"GLO12 matched {r['n_matched_glo12']} weekly dates")

        suffix = "" if season == "winter" else f"_{season}"
        npz_path = NPZ_DIR / f"member_spectra_leadtime{leadtime}{suffix}.npz"
        np.savez(
            npz_path,
            freq_r=r["freq_r"],
            psd_members=r["psd_members"],
            psd_det=r["psd_det"],
            psd_glo12=r["psd_glo12"] if r["psd_glo12"] is not None else np.array([]),
            n_matched=r["n_matched"],
            n_matched_glo12=r["n_matched_glo12"],
        )
        print(f"  Saved raw PSDs: {npz_path}")

        fig = plot_member_spectra(
            leadtime, r["freq_r"], r["psd_members"], r["psd_det"], r["n_matched"],
            r["psd_glo12"], r["n_matched_glo12"], season,
        )
        out_path = OUT_DIR / f"member_spectra_fm_unet_crps10_leadtime{leadtime}{suffix}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--leadtimes", type=int, nargs="+", default=[0, 3, 5])
    parser.add_argument("--season", choices=sorted(SEASON_RANGES), default="winter")
    args = parser.parse_args()
    main(args.leadtimes, args.season)
