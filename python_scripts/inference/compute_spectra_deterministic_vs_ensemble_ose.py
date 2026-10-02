"""
Gulf Stream box wavenumber spectra for the OSE FM UNet (DUACS target, real L3
input), same diagnostic as compute_spectra_deterministic_vs_ensemble_crps10.py
(left untouched): deterministic UNet vs FM ensemble mean (+/- 1 std) vs GLO12,
plus a figure overlaying the OSE and OSSE (crps10) ensemble means.

The box, PSD estimator, GLO12 matching/regridding and plot style are imported
from that script; only the data loading differs: forecasts are read from the
per-lead-time files test_leadtime_<14+lt>.nc (time = valid date), since the
OSE eval folder keeps those instead of the per-window files.

CPU only, no inference.

Usage:
    python compute_spectra_deterministic_vs_ensemble_ose.py [--leadtimes 0 3 5]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_spectra_deterministic_vs_ensemble_crps10 import (  # noqa: E402
    DET_DIR, GLO12_CROP_PAD, GLO12_LAT, GLO12_LON, LAT_BOUNDS, LON_BOUNDS, OUT_DIR,
    build_glo12_date_index, isotropic_psd, plot_spectra, regrid_to_target,
)

OUTPUTS = Path("/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/")
RUNS = {
    "OSE": OUTPUTS / "eval_nrt2023_sla_filtered_90d_ose_duacs",
    "OSSE": OUTPUTS / "eval_nrt2023_fm_unet_crps10",
}


def box_masks(lat, lon):
    latmask = (lat >= LAT_BOUNDS[0]) & (lat <= LAT_BOUNDS[1])
    lonmask = (lon >= LON_BOUNDS[0]) & (lon <= LON_BOUNDS[1])
    return latmask, lonmask, lat[latmask], lon[lonmask]


def compute_mean_spectra(eval_dir, leadtime):
    ds = xr.open_dataset(eval_dir / f"test_leadtime_{14 + leadtime}.nc")
    ds_det = xr.open_dataset(DET_DIR / f"test_data_{14 + leadtime}.nc")
    glo12_index = build_glo12_date_index(leadtime)
    g_latmask = (GLO12_LAT >= LAT_BOUNDS[0] - GLO12_CROP_PAD) & (GLO12_LAT <= LAT_BOUNDS[1] + GLO12_CROP_PAD)
    g_lonmask = (GLO12_LON >= LON_BOUNDS[0] - GLO12_CROP_PAD) & (GLO12_LON <= LON_BOUNDS[1] + GLO12_CROP_PAD)

    latmask, lonmask, sub_lat, sub_lon = box_masks(ds.lat.values, ds.lon.values)
    psd = {k: [] for k in ("det", "mean", "plus", "minus", "glo12")}
    freq_r = None
    for t in ds.time.values:
        try:
            det_box = ds_det["out"].sel(time=t).values[np.ix_(latmask, lonmask)]
        except KeyError:
            continue
        day = ds.sel(time=t)
        mean_box = day.forecast_mean.values[np.ix_(latmask, lonmask)]
        std_box = day.forecast_std.values[np.ix_(latmask, lonmask)]
        iso = isotropic_psd(det_box, sub_lat, sub_lon)
        freq_r = iso["freq_r"].values if freq_r is None else freq_r
        psd["det"].append(iso.values)
        psd["mean"].append(isotropic_psd(mean_box, sub_lat, sub_lon).values)
        psd["plus"].append(isotropic_psd(mean_box + std_box, sub_lat, sub_lon).values)
        psd["minus"].append(isotropic_psd(mean_box - std_box, sub_lat, sub_lon).values)

        glo12_path = glo12_index.get(np.datetime_as_string(t, unit="D"))
        if glo12_path is not None:
            with xr.open_dataset(glo12_path) as g:
                zos = g["zos"].isel(time=0).values[np.ix_(g_latmask, g_lonmask)]
            zos = regrid_to_target(zos, GLO12_LAT[g_latmask], GLO12_LON[g_lonmask], sub_lat, sub_lon)
            psd["glo12"].append(isotropic_psd(zos, sub_lat, sub_lon).values)
    ds.close()
    ds_det.close()
    n, n_g = len(psd["mean"]), len(psd["glo12"])
    print(f"  {eval_dir.name} leadtime={leadtime}: {n} days, GLO12 {n_g} weeks", flush=True)
    mean = {k: (np.nanmean(v, axis=0) if v else None) for k, v in psd.items()}
    return freq_r, mean, n, n_g


def plot_ose_vs_osse(leadtime, freq_r, res):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"OSE": "#1b9e77", "OSSE": "#2a78d6"}
    for name, (m, n) in res.items():
        ax.fill_between(freq_r, m["minus"], m["plus"], color=colors[name], alpha=0.15)
        ax.plot(freq_r, m["mean"], color=colors[name], linewidth=2,
                label=f"FM UNet {name} ensemble mean ($\\pm$ 1 std), {n} days")
    m_ose = res["OSE"][0]
    ax.plot(freq_r, m_ose["det"], color="#eb6834", linewidth=2, linestyle="--", label="Deterministic UNet")
    if m_ose["glo12"] is not None:
        ax.plot(freq_r, m_ose["glo12"], color="#2a2a2a", linewidth=2, linestyle=":",
                label="GLO12 (CMEMS forecast, regridded)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber [cycles/km]")
    ax.set_ylabel("PSD [m$^2$/(cycles/km)]")
    ax.set_title(f"Gulf Stream box wavenumber spectra -- leadtime {leadtime}d\n"
                 f"FM UNet OSE vs OSSE, 2023 NRT")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.3)
    fig.tight_layout()
    return fig


def main(leadtimes):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for lt in leadtimes:
        res, freq_r, n_g_ose = {}, None, 0
        for name, eval_dir in RUNS.items():
            freq_r, m, n, n_g = compute_mean_spectra(eval_dir, lt)
            res[name] = (m, n)
            if name == "OSE":
                n_g_ose = n_g
        m, n = res["OSE"]
        fig = plot_spectra(lt, freq_r, m["det"], m["mean"], m["plus"], m["minus"], n, m["glo12"], n_g_ose)
        fig.axes[0].set_title(fig.axes[0].get_title().replace("ensemble FM UNet", "ensemble FM UNet OSE"))
        out = OUT_DIR / f"spectra_deterministic_vs_ensemble_ose_leadtime{lt}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", out, flush=True)
        fig = plot_ose_vs_osse(lt, freq_r, res)
        out = OUT_DIR / f"spectra_fm_unet_ose_vs_osse_leadtime{lt}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--leadtimes", type=int, nargs="+", default=[0, 3, 5])
    main(parser.parse_args().leadtimes)
