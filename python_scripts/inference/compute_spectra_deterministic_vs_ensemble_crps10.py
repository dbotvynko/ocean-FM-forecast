"""
Compare wavenumber power spectra of the deterministic UNet SLA reconstruction
against the FM UNet (crps10) ensemble forecast, over the Gulf Stream box --
the standard regional diagnostic used in this group's SSH-mapping papers to
check whether a model is over-smoothed (missing small-scale/high-wavenumber
energy) relative to a more realistic field.

Data:
  - Deterministic: /Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_UNet_Filtered/
    nrt_sla/test_data_{14+leadtime}.nc ("out", dense, full 2023 daily record).
    The "14+leadtime" offset is inferred from forecast_DDPM_UNet_1patch.yaml's
    patch_dims.time=29 (14 context days, then leadtime 0..6 forecast days) and
    confirmed directly: for init_time=2023-01-01 in the crps10 eval files,
    leadtime=0's valid_time is 2023-01-15 -- a 14-day offset.
  - Ensemble: the same eval_nrt2023_fm_unet_crps10/*.nc files used elsewhere
    (forecast_mean, forecast_std), matched to the deterministic field by exact
    calendar date (valid_time).
  - GLO12: CMEMS Mercator GLO12 operational forecast, from the local weekly
    archive at /Odyssey/public/glorys/mercator_forecast/glo12/R2023*/ (52
    weekly init dates covering all of 2023, each with 14 daily forecast files
    "glo12_rg_1d-m_{valid_date}-{valid_date}_fcst_R{init_date}.nc"). For a
    given leadtime, valid_date = init_date + leadtime; this reimplements (self
    -contained, to avoid the heavy torch/pyinterp imports at the top of that
    module) the same logic as get_preprocessed_rec_mercator_forecast() in
    FORECAST/2023a_SSH_mapping_OSE/.../src/mod_intervals.py. GLO12 is only
    available at weekly cadence, so it's matched against whichever of the
    ~90 ensemble/deterministic dates happen to coincide (checked: 13/leadtime,
    every ~1-in-7 days as expected) -- a smaller but still meaningful sample.
    GLO12 is on its own native 1/12deg grid (2041x4320, hardcoded lat/lon
    since the files carry no lat/lon coordinates), independent of the
    ~0.25deg grid shared by the deterministic/ensemble fields. Comparing raw
    PSDs across the two grids would conflate real spectral differences with
    a resolution artifact (the finer grid trivially carries more
    high-wavenumber power), so GLO12 is bilinearly regridded onto the
    deterministic/ensemble lat/lon grid before computing its PSD -- all
    three curves then share the same freq_r wavenumber bins.

We do NOT include the sparse along-track "truth" here: it's real altimetry,
only ~10-20% spatially covered on any given day (checked directly), so a
naive 2D-FFT PSD on it would be dominated by the sampling mask rather than
the ocean signal. GLO12 -- a dense operational analysis/forecast -- serves as
the actual reference field for this comparison instead.

Since only forecast_mean/forecast_std are saved (not individual ensemble
members), the "ensemble spread" is shown as the spectrum of forecast_mean
plus and minus one forecast_std, shading the region between them -- the
standard mean +/- spread envelope, not literal per-member spectra.

Usage:
    python compute_spectra_deterministic_vs_ensemble_crps10.py [--leadtimes 0 3 5]
"""

import argparse
import datetime
import glob
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xrft

# xrft warns once per call that the outermost isotropic bin can exceed the
# Nyquist wavenumber -- expected and harmless for our box size/nfactor, and
# would otherwise print ~4 times per matched day (over 300 times per leadtime).
warnings.filterwarnings("ignore", message="Isotropic wavenumber larger than the Nyquist wavenumber.*")

DATA_DIR = Path("/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/eval_nrt2023_fm_unet_crps10/")
DET_DIR = Path("/Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_UNet_Filtered/nrt_sla/")
GLO12_GLOB = "/Odyssey/public/glorys/mercator_forecast/glo12/R2023*/"
OUT_DIR = Path("/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/figures/")

# GLO12 files carry no lat/lon coordinates -- same fixed 1/12deg grid used in
# get_preprocessed_rec_mercator_forecast() (mod_intervals.py).
GLO12_LAT = np.linspace(-80, 90, 2041)
GLO12_LON = np.linspace(-180, 180, 4320)

# Gulf Stream box: the standard region for this kind of spectral/effective-
# resolution diagnostic (same box used in plot_variance_maps_fm_unet_crps10.py
# and metrics_domains.yaml); checked to be 100% ocean (no land pixels ever
# without a valid truth observation across the 2023 record).
LAT_BOUNDS = (32, 42)
LON_BOUNDS = (-65, -55)
EARTH_RADIUS_KM = 6371.0

# Extra margin (deg) when cropping the finer GLO12 grid, so the crop always
# fully brackets the coarser target grid and .interp() never has to
# extrapolate at the box edges.
GLO12_CROP_PAD = 0.5


def box_to_km_dataarray(box: np.ndarray, sub_lat: np.ndarray, sub_lon: np.ndarray) -> xr.DataArray:
    lat0 = sub_lat.mean()
    x_km = (sub_lon - sub_lon.mean()) * np.pi / 180 * EARTH_RADIUS_KM * np.cos(np.deg2rad(lat0))
    y_km = (sub_lat - sub_lat.mean()) * np.pi / 180 * EARTH_RADIUS_KM
    return xr.DataArray(box, coords={"y": y_km, "x": x_km}, dims=["y", "x"])


def regrid_to_target(box: np.ndarray, src_lat: np.ndarray, src_lon: np.ndarray,
                      target_lat: np.ndarray, target_lon: np.ndarray) -> np.ndarray:
    """Bilinearly interpolate a (finer-grid) box onto the target lat/lon grid."""
    da = xr.DataArray(box, coords={"lat": src_lat, "lon": src_lon}, dims=["lat", "lon"])
    return da.interp(lat=target_lat, lon=target_lon, method="linear").values


def isotropic_psd(box: np.ndarray, sub_lat: np.ndarray, sub_lon: np.ndarray) -> xr.DataArray:
    # A single NaN pixel is enough to make the whole FFT come back NaN (hit
    # this with GLO12: one tiny coastal island pixel in an otherwise
    # 100%-ocean box, and occasional edge NaNs from regridding). Gap-fill
    # with the box mean -- negligible effect on the spectrum given it's at
    # most a handful of pixels.
    if np.isnan(box).any():
        box = np.where(np.isnan(box), np.nanmean(box), box)
    da = box_to_km_dataarray(box, sub_lat, sub_lon)
    return xrft.isotropic_power_spectrum(
        da, dim=["x", "y"], detrend="linear", window="hann", scaling="density", nfactor=4
    )


def build_glo12_date_index(leadtime: int) -> dict:
    """Map valid-date string -> GLO12 file path, for this leadtime, across
    all 52 weekly 2023 init folders."""
    index = {}
    for folder in sorted(glob.glob(GLO12_GLOB)):
        folder = folder.rstrip("/")
        r_name = folder.split("/")[-1]
        init_date = datetime.datetime.strptime(r_name[1:9], "%Y%m%d")
        valid_date = init_date + datetime.timedelta(days=leadtime)
        date_str = valid_date.strftime("%Y%m%d")
        fpath = Path(folder) / f"glo12_rg_1d-m_{date_str}-{date_str}_fcst_{r_name}.nc"
        if fpath.exists():
            index[valid_date.strftime("%Y-%m-%d")] = fpath
    return index


def compute_mean_spectra(leadtime: int):
    det_file = DET_DIR / f"test_data_{14 + leadtime}.nc"
    ds_det = xr.open_dataset(det_file)
    glo12_index = build_glo12_date_index(leadtime)

    glo12_latmask = (GLO12_LAT >= LAT_BOUNDS[0] - GLO12_CROP_PAD) & (GLO12_LAT <= LAT_BOUNDS[1] + GLO12_CROP_PAD)
    glo12_lonmask = (GLO12_LON >= LON_BOUNDS[0] - GLO12_CROP_PAD) & (GLO12_LON <= LON_BOUNDS[1] + GLO12_CROP_PAD)
    glo12_sub_lat = GLO12_LAT[glo12_latmask]
    glo12_sub_lon = GLO12_LON[glo12_lonmask]

    eval_files = sorted(DATA_DIR.glob("*.nc"))
    psd_det, psd_mean, psd_plus, psd_minus, psd_glo12 = [], [], [], [], []
    freq_r = None
    n_matched = 0
    n_matched_glo12 = 0
    n_skipped_leadtime = 0
    n_skipped_date = 0

    lat = lon = None
    for f in eval_files:
        with xr.open_dataset(f) as ds:
            if "leadtime" not in ds.sizes or leadtime not in range(ds.sizes["leadtime"]):
                n_skipped_leadtime += 1
                continue
            if lat is None:
                lat = ds["lat"].values
                lon = ds["lon"].values
                latmask = (lat >= LAT_BOUNDS[0]) & (lat <= LAT_BOUNDS[1])
                lonmask = (lon >= LON_BOUNDS[0]) & (lon <= LON_BOUNDS[1])
                sub_lat = lat[latmask]
                sub_lon = lon[lonmask]

            valid_time = ds["valid_time"].isel(leadtime=leadtime).values
            try:
                det_slice = ds_det["out"].sel(time=valid_time).values
            except KeyError:
                n_skipped_date += 1
                continue

            fmean = ds["forecast_mean"].isel(leadtime=leadtime).values
            fstd = ds["forecast_std"].isel(leadtime=leadtime).values

            det_box = det_slice[np.ix_(latmask, lonmask)]
            mean_box = fmean[np.ix_(latmask, lonmask)]
            std_box = fstd[np.ix_(latmask, lonmask)]

            iso = isotropic_psd(det_box, sub_lat, sub_lon)
            if freq_r is None:
                freq_r = iso["freq_r"].values
            psd_det.append(iso.values)
            psd_mean.append(isotropic_psd(mean_box, sub_lat, sub_lon).values)
            psd_plus.append(isotropic_psd(mean_box + std_box, sub_lat, sub_lon).values)
            psd_minus.append(isotropic_psd(mean_box - std_box, sub_lat, sub_lon).values)
            n_matched += 1

            date_str = np.datetime_as_string(valid_time, unit="D")
            glo12_path = glo12_index.get(date_str)
            if glo12_path is not None:
                with xr.open_dataset(glo12_path) as ds_glo12:
                    zos = ds_glo12["zos"].isel(time=0).values
                glo12_box = zos[np.ix_(glo12_latmask, glo12_lonmask)]
                glo12_box = regrid_to_target(glo12_box, glo12_sub_lat, glo12_sub_lon, sub_lat, sub_lon)
                psd_glo12.append(isotropic_psd(glo12_box, sub_lat, sub_lon).values)
                n_matched_glo12 += 1

    ds_det.close()
    print(f"  leadtime={leadtime}: matched {n_matched} days "
          f"(skipped {n_skipped_leadtime} without this leadtime, "
          f"{n_skipped_date} with no matching deterministic date); "
          f"GLO12 matched {n_matched_glo12}/{len(glo12_index)} weekly dates")

    return (
        freq_r,
        np.nanmean(psd_det, axis=0),
        np.nanmean(psd_mean, axis=0),
        np.nanmean(psd_plus, axis=0),
        np.nanmean(psd_minus, axis=0),
        n_matched,
        np.nanmean(psd_glo12, axis=0) if psd_glo12 else None,
        n_matched_glo12,
    )


def plot_spectra(leadtime, freq_r, psd_det, psd_mean, psd_plus, psd_minus, n_matched,
                  psd_glo12, n_matched_glo12) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.fill_between(freq_r, psd_minus, psd_plus, color="#2a78d6", alpha=0.2,
                     label="Ensemble mean $\\pm$ 1 std")
    ax.plot(freq_r, psd_mean, color="#2a78d6", linewidth=2, label="Ensemble mean (FM UNet)")
    ax.plot(freq_r, psd_det, color="#eb6834", linewidth=2, linestyle="--", label="Deterministic UNet")
    if psd_glo12 is not None:
        ax.plot(freq_r, psd_glo12, color="#2a2a2a", linewidth=2, linestyle=":",
                 label=f"GLO12 (CMEMS forecast, {n_matched_glo12} weeks, regridded)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber [cycles/km]")
    ax.set_ylabel("PSD [m$^2$/(cycles/km)]")
    ax.set_title(
        f"Gulf Stream box wavenumber spectra -- leadtime {leadtime}d\n"
        f"deterministic vs. ensemble FM UNet vs. GLO12, 2023 NRT mean over {n_matched} days"
    )
    ax.legend(frameon=False)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.3)
    fig.tight_layout()
    return fig


def main(leadtimes):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for leadtime in leadtimes:
        (freq_r, psd_det, psd_mean, psd_plus, psd_minus, n_matched,
         psd_glo12, n_matched_glo12) = compute_mean_spectra(leadtime)
        fig = plot_spectra(leadtime, freq_r, psd_det, psd_mean, psd_plus, psd_minus, n_matched,
                            psd_glo12, n_matched_glo12)
        out_path = OUT_DIR / f"spectra_deterministic_vs_ensemble_crps10_leadtime{leadtime}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--leadtimes", type=int, nargs="+", default=[0, 3, 5])
    args = parser.parse_args()
    main(args.leadtimes)
