"""
Redraw the per-member Gulf Stream spectra figures of
compute_member_spectra_fm_unet_crps10.py with the ensemble-mean spectrum
added, saved as separate "_with_mean" figures so the originals are kept.

The ensemble-mean curve is the spectrum of forecast_mean (the saved
10-member mean field), read from eval_nrt_2023_fm_unet_crps10.py's per-day
.nc files over the same season window -- the same quantity as the
"Ensemble mean (FM UNet)" curve of
compute_spectra_deterministic_vs_ensemble_crps10.py. Averaging the fields
cancels the members' uncorrelated small-scale structure, so it drops below
the member curves at high wavenumber.

CPU only, no inference rerun.

Usage:
    python plot_member_spectra_with_mean_fm_unet_crps10.py [--leadtimes 0 3 5] [--season winter|summer]
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", message="Isotropic wavenumber larger than the Nyquist wavenumber.*")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_spectra_deterministic_vs_ensemble_crps10 import LAT_BOUNDS, LON_BOUNDS, isotropic_psd  # noqa: E402

OUTPUTS = Path("/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/")
NPZ_DIR = OUTPUTS / "member_spectra_crps10"
OUT_DIR = OUTPUTS / "figures"
EVAL_DIRS = {"winter": OUTPUTS / "eval_nrt2023_fm_unet_crps10", "summer": OUTPUTS / "eval_nrt2023_fm_unet_crps10_summer"}
SEASON_RANGES = {"winter": ("2023-01-01", "2023-03-31"), "summer": ("2023-07-01", "2023-09-28")}


def ensemble_mean_field_psd(leadtimes, season):
    """Day-averaged PSD of forecast_mean in the Gulf Stream box, per leadtime."""
    start, end = SEASON_RANGES[season]
    psds = {lt: [] for lt in leadtimes}
    for d in pd.date_range(start, end, freq="D"):
        f = EVAL_DIRS[season] / f"{d.date()}.nc"
        if not f.exists():
            continue
        with xr.open_dataset(f) as ds:
            box = ds["forecast_mean"].sel(lat=slice(*LAT_BOUNDS), lon=slice(*LON_BOUNDS))
            for lt in leadtimes:
                if lt in ds["leadtime"].values:
                    b = box.sel(leadtime=lt)
                    psds[lt].append(isotropic_psd(b.values, b.lat.values, b.lon.values).values)
    return {lt: (np.nanmean(p, axis=0) if p else None, len(p)) for lt, p in psds.items()}


def plot(leadtime, npz, psd_mean_field, n_mean_field, season):
    freq_r = npz["freq_r"]
    psd_members = npz["psd_members"]  # (n_days, num_samples, nfreq)
    num_samples = psd_members.shape[1]
    member_curves = np.nanmean(psd_members, axis=0)  # (num_samples, nfreq)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.95, num_samples))
    for m in range(num_samples):
        ax.plot(freq_r, member_curves[m], color=colors[m], linewidth=1.2, alpha=0.85,
                label=f"Ensemble members ({num_samples})" if m == 0 else None)

    if psd_mean_field is not None:
        ax.plot(freq_r, psd_mean_field, color="#c51b8a", linewidth=2.2,
                label=f"Ensemble mean (FM UNet, {n_mean_field} days)")

    ax.plot(freq_r, npz["psd_det"], color="#eb6834", linewidth=2, linestyle="--", label="Deterministic UNet")
    if npz["psd_glo12"].size:
        ax.plot(freq_r, npz["psd_glo12"], color="#2a2a2a", linewidth=2, linestyle=":",
                label=f"GLO12 (CMEMS forecast, {int(npz['n_matched_glo12'])} weeks, regridded)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber [cycles/km]")
    ax.set_ylabel("PSD [m$^2$/(cycles/km)]")
    ax.set_title(
        f"Gulf Stream box wavenumber spectra per member -- leadtime {leadtime}d\n"
        f"FM UNet ensemble vs. deterministic UNet vs. GLO12, 2023 {season} NRT mean over {int(npz['n_matched'])} days"
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.3)
    fig.tight_layout()
    return fig


def main(leadtimes, season):
    suffix = "" if season == "winter" else f"_{season}"
    mean_field = ensemble_mean_field_psd(leadtimes, season)
    for lt in leadtimes:
        npz = np.load(NPZ_DIR / f"member_spectra_leadtime{lt}{suffix}.npz")
        psd_mf, n_mf = mean_field[lt]
        fig = plot(lt, npz, psd_mf, n_mf, season)
        out_path = OUT_DIR / f"member_spectra_fm_unet_crps10_leadtime{lt}{suffix}_with_mean.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved ({n_mf} days for ensemble-mean field):", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--leadtimes", type=int, nargs="+", default=[0, 3, 5])
    parser.add_argument("--season", choices=sorted(SEASON_RANGES), default="winter")
    args = parser.parse_args()
    main(args.leadtimes, args.season)
