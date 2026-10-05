"""
Per-member Gulf Stream wavenumber spectra of the 2024 OceanBench FM UNet ensembles and of their hybrid versions,
whose members are the deterministic Nadir-only UNet plus each FM member's deviation from the FM ensemble mean:
    hybrid_i = UNet + (member_i - mean(members))

Reads the member files written by eval_2024_oceanbench_fm_unet_members.py (no GPU). Same box and isotropic PSD as
compute_member_spectra_fm_unet_crps10.py; GLO12 forecasts added where the 2024 files exist. Spectra are computed per
forecast start and averaged over the 48 Wednesday starts.

Usage:
    python compute_member_spectra_hybrid_2024.py TAG [TAG ...] [--leadtimes 0 2 4 6]
      TAG: suffix of outputs/eval_2024_oceanbench_members_<TAG>/ (e.g. ose_ep233 osse_ep153)
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_spectra_deterministic_vs_ensemble_crps10 import (  # noqa: E402
    GLO12_CROP_PAD, GLO12_LAT, GLO12_LON, LAT_BOUNDS, LON_BOUNDS, isotropic_psd, regrid_to_target,
)

OUTPUTS = Path("/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/")
UNET_DIR = Path("/Odyssey/public/glorys/rec/forecast_Unet_SLA_1patch_OSSE_NadirOnly_inp_NadirNoSwonOnlyCLS_eval2024/nrt_sla/")
GLO12_ROOT = Path("/Odyssey/public/glorys/mercator_forecast/glo12/")
WEDNESDAYS = pd.date_range("2024-01-17", "2024-12-11", freq="7D")
OBS_DAYS = 14
LABELS = {"ose_ep233": "OSE", "osse_ep153": "OSSE"}


def glo12_box(valid: pd.Timestamp, init: pd.Timestamp, sub_lat, sub_lon):
    """GLO12 forecast started on `init` (the Wednesday), valid on `valid`, regridded onto the box; None if missing."""
    folder = GLO12_ROOT / f"R{init:%Y%m%d}"
    path = folder / f"glo12_rg_1d-m_{valid:%Y%m%d}-{valid:%Y%m%d}_fcst_R{init:%Y%m%d}.nc"
    if not path.exists():
        return None
    latmask = (GLO12_LAT >= LAT_BOUNDS[0] - GLO12_CROP_PAD) & (GLO12_LAT <= LAT_BOUNDS[1] + GLO12_CROP_PAD)
    lonmask = (GLO12_LON >= LON_BOUNDS[0] - GLO12_CROP_PAD) & (GLO12_LON <= LON_BOUNDS[1] + GLO12_CROP_PAD)
    with xr.open_dataset(path) as ds:
        zos = ds["zos"].isel(time=0).values
    return regrid_to_target(zos[np.ix_(latmask, lonmask)], GLO12_LAT[latmask], GLO12_LON[lonmask], sub_lat, sub_lon)


def spectra(tag: str, leadtimes: list[int]) -> dict:
    """{lt: dict(freq_r, fm (starts, members, nf), hybrid (same), fm_mean, unet, glo12 (starts, nf))}."""
    members_dir = OUTPUTS / f"eval_2024_oceanbench_members_{tag}"
    unet = {lt: xr.open_dataset(UNET_DIR / f"test_data_{OBS_DAYS + lt}.nc")["out"] for lt in leadtimes}
    out = {lt: {k: [] for k in ("fm", "hybrid", "fm_mean", "unet", "glo12")} for lt in leadtimes}
    for wednesday in WEDNESDAYS:
        ds = xr.open_dataset(members_dir / f"{(wednesday - pd.Timedelta(days=OBS_DAYS)).date()}.nc")
        latmask = (ds.lat.values >= LAT_BOUNDS[0]) & (ds.lat.values <= LAT_BOUNDS[1])
        lonmask = (ds.lon.values >= LON_BOUNDS[0]) & (ds.lon.values <= LON_BOUNDS[1])
        sub_lat, sub_lon = ds.lat.values[latmask], ds.lon.values[lonmask]
        for lt in leadtimes:
            valid = wednesday + pd.Timedelta(days=lt)
            members = ds.forecast.sel(leadtime=lt).values[:, latmask][:, :, lonmask]
            mean = members.mean(axis=0)
            u = unet[lt].sel(time=valid).sel(lat=sub_lat, lon=sub_lon, method="nearest").values
            psd = lambda field: isotropic_psd(field, sub_lat, sub_lon)
            out[lt]["freq_r"] = psd(mean)["freq_r"].values
            out[lt]["fm"].append([psd(m).values for m in members])
            out[lt]["hybrid"].append([psd(u + m - mean).values for m in members])
            out[lt]["fm_mean"].append(psd(mean).values)
            out[lt]["unet"].append(psd(u).values)
            g = glo12_box(valid, wednesday, sub_lat, sub_lon)
            if g is not None:
                out[lt]["glo12"].append(psd(g).values)
        print(tag, wednesday.date(), flush=True)
    for lt in leadtimes:
        for k in ("fm", "hybrid", "fm_mean", "unet", "glo12"):
            out[lt][k] = np.array(out[lt][k])
        np.savez(OUTPUTS / f"member_spectra_hybrid_2024_{tag}_leadtime{lt}.npz", **out[lt])
    return out


def band(ax, freq, per_start_members, color, label):
    per_member = per_start_members.mean(axis=0)  # (members, nf), averaged over starts
    ax.fill_between(freq, per_member.min(0), per_member.max(0), color=color, alpha=0.2)
    ax.plot(freq, np.median(per_member, axis=0), color=color, linewidth=1.8, label=label)


def plot(results: dict, leadtimes: list[int]) -> None:
    colors = {"ose_ep233": ("#1baf7a", "#4a3aa7"), "osse_ep153": ("#2a78d6", "#e87ba4")}
    fig, axes = plt.subplots(1, len(leadtimes), figsize=(4.6 * len(leadtimes), 4.8), sharey=True)
    for ax, lt in zip(np.atleast_1d(axes), leadtimes):
        first = next(iter(results.values()))[lt]
        freq = first["freq_r"]
        for tag, r in results.items():
            fm_color, hybrid_color = colors.get(tag, ("#888888", "#444444"))
            name = LABELS.get(tag, tag)
            band(ax, freq, r[lt]["fm"], fm_color, f"FM {name} members")
            band(ax, freq, r[lt]["hybrid"], hybrid_color, f"Hybrid {name} members (UNet + FM {name} deviation)")
            ax.plot(freq, r[lt]["fm_mean"].mean(0), color=fm_color, linewidth=1.4, linestyle="--", label=f"FM {name} ensemble mean")
        ax.plot(freq, first["unet"].mean(0), color="#eb6834", linewidth=2, linestyle="--", label="Deterministic UNet (Nadir only)")
        if len(first["glo12"]):
            ax.plot(freq, first["glo12"].mean(0), color="#2a2a2a", linewidth=1.8, linestyle=":", label=f"GLO12 forecast ({len(first['glo12'])} starts)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Wavenumber [cycles/km]")
        ax.set_title(f"Lead day {lt + 1}")
        ax.grid(True, which="both", linewidth=0.4, alpha=0.3)
        secondary = ax.secondary_xaxis("top", functions=(lambda k: 1 / np.maximum(k, 1e-9), lambda w: 1 / np.maximum(w, 1e-9)))
        secondary.set_xlabel("Wavelength [km]")
    np.atleast_1d(axes)[0].set_ylabel("PSD [m$^2$/(cycles/km)]")
    np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle(f"Gulf Stream box ({LAT_BOUNDS[0]}–{LAT_BOUNDS[1]}°N, {-LON_BOUNDS[0]}–{-LON_BOUNDS[1]}°W) SLA spectra per member, "
                 f"48 OceanBench 2024 starts; bands: min–max over members, line: median member", fontsize=11)
    fig.tight_layout()
    out = OUTPUTS / "figures" / f"member_spectra_hybrid_2024_{'_'.join(results)}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("Saved:", out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tags", nargs="+")
    parser.add_argument("--leadtimes", type=int, nargs="+", default=[0, 2, 4, 6])
    args = parser.parse_args()
    results = {tag: spectra(tag, args.leadtimes) for tag in args.tags}
    plot(results, args.leadtimes)
