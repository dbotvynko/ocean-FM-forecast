"""
Plot ensemble-variance maps (Var[forecast] = forecast_std**2) for the FM UNet
(forecast_DDPM_UNet_1patch) 2023 NRT crps10 evaluation, with cartopy.

Reads the per-day .nc files saved by eval_nrt_2023_fm_unet_crps10.py (each has
forecast_std(leadtime, lat, lon)), averages forecast_std**2 over all available
days for a given leadtime, masks out land (using "truth" SLA's own NaN
pattern, since forecast_std is otherwise defined everywhere including land),
and plots:
  - one global map per leadtime (no domain-box overlays -- kept only below).
  - one zoomed-in panel per domain (Gulfstream/Groenland/Madere, from
    forecast/notebooks/metrics_domains.yaml), each bordered with a plain
    sharp-cornered rectangle (matplotlib.patches.Rectangle -- never rounded,
    unlike a FancyBboxPatch/boxstyle="round" box).

Also plots the same maps normalized by the temporal variance of the "truth"
SLA field itself (the L3 altimetry reference, same field used as the
EKE-proxy denominator in check_variance_correlations_fm_unet_crps10.py):
ratio = forecast_variance / truth_variance. Raw ensemble variance is hard to
judge in isolation -- a few cm^2 means very different things in a
low-mesoscale-activity region vs. the Gulf Stream -- so normalizing by the
local natural SLA variance gives a scale-free measure of how much of the
truth's own variability the ensemble spread represents.

Usage:
    python plot_variance_maps_fm_unet_crps10.py [--leadtimes 0 3 5] [--season winter|summer]
"""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# --season picks which eval_nrt_2023_fm_unet_crps10.py output folder to read;
# summer figures get a "_summer" filename suffix so winter ones are kept.
DATA_DIRS = {
    "winter": Path("/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/eval_nrt2023_fm_unet_crps10/"),
    "summer": Path("/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/eval_nrt2023_fm_unet_crps10_summer/"),
}
OUT_DIR = Path("/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/figures/")

# Same regions as forecast/notebooks/metrics_domains.yaml (lon/lat given as
# [min, max] here instead of a builtins.slice, everything else identical).
DOMAINS = {
    "Gulfstream": {"lon": (-65, -55), "lat": (32, 42), "color": "#e07b39"},
    "Groenland": {"lon": (-35, -25), "lat": (48, 58), "color": "#4c72b0"},
    "Madere": {"lon": (-23, -13), "lat": (32, 42), "color": "#55a868"},
}


def load_mean_variance(leadtime: int, data_dir: Path, return_counts: bool = False):
    """Returns (mean_variance, truth_variance): the ensemble's mean forecast
    variance, and the temporal variance of the "truth" L3 SLA field itself
    (same quantity as check_variance_correlations_fm_unet_crps10.py's
    EKE-proxy denominator), both land-masked. With return_counts=True also
    returns the raw per-pixel truth moments (count, sum, sum of squares), so
    callers can pool them spatially."""
    files = sorted(data_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No .nc files found in {data_dir}")

    # Not open_mfdataset(combine="nested", ...): per-file "leadtime" values
    # aren't consistent across all days (some windows are short a leadtime
    # near the edges of the 2023 record), which makes xarray's strict
    # coordinate alignment fail. We only need the numeric mean, so accumulate
    # it by hand instead -- skip any file missing this leadtime index.
    # Also accumulate "truth" validity/moments: the model outputs
    # forecast_std everywhere including land (checked directly --
    # 881921/881921 land pixels have a finite forecast_std in a sample
    # file), but "truth" SLA is properly NaN over land. We reuse that as a
    # land mask, and its first/second moments to get its temporal variance.
    running_sum = None
    count = None
    sum_truth = sum_truth2 = truth_count = None
    lat = lon = None
    skipped = []
    for f in files:
        with xr.open_dataset(f) as ds:
            if "leadtime" not in ds.sizes or leadtime not in range(ds.sizes["leadtime"]):
                skipped.append(f.name)
                continue
            std = ds["forecast_std"].isel(leadtime=leadtime).values
            truth = ds["truth"].isel(leadtime=leadtime).values
            if lat is None:
                lat = ds["lat"].values
                lon = ds["lon"].values
                running_sum = np.zeros_like(std, dtype=np.float64)
                count = np.zeros_like(std, dtype=np.float64)
                sum_truth = np.zeros_like(std, dtype=np.float64)
                sum_truth2 = np.zeros_like(std, dtype=np.float64)
                truth_count = np.zeros_like(std, dtype=np.float64)
            valid = np.isfinite(std)
            running_sum[valid] += (std[valid].astype(np.float64)) ** 2
            count[valid] += 1
            tvalid = np.isfinite(truth)
            sum_truth[tvalid] += truth[tvalid].astype(np.float64)
            sum_truth2[tvalid] += truth[tvalid].astype(np.float64) ** 2
            truth_count[tvalid] += 1

    if skipped:
        print(f"  (leadtime={leadtime}: skipped {len(skipped)}/{len(files)} files without it, "
              f"e.g. {skipped[:3]})")

    if lat is None:
        raise RuntimeError(f"No file had a valid leadtime={leadtime} slice")

    with np.errstate(invalid="ignore"):
        mean_variance = np.where(count > 0, running_sum / count, np.nan)
        mean_truth = np.where(truth_count > 0, sum_truth / truth_count, np.nan)
        mean_truth2 = np.where(truth_count > 0, sum_truth2 / truth_count, np.nan)
        truth_variance = mean_truth2 - mean_truth ** 2
    # Land mask: a pixel is ocean only if "truth" was valid at least once.
    mean_variance = np.where(truth_count > 0, mean_variance, np.nan)
    truth_variance = np.where(truth_count > 0, truth_variance, np.nan)

    coords = {"lat": lat, "lon": lon}
    out = (
        xr.DataArray(mean_variance, coords=coords, dims=["lat", "lon"]),
        xr.DataArray(truth_variance, coords=coords, dims=["lat", "lon"]),
    )
    if return_counts:
        out += tuple(xr.DataArray(a, coords=coords, dims=["lat", "lon"]) for a in (truth_count, sum_truth, sum_truth2))
    return out


def plot_global_map(variance: xr.DataArray, leadtime: int, vmax: float, cbar_label: str, title: str) -> plt.Figure:
    fig, ax = plt.subplots(
        figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    mesh = ax.pcolormesh(
        variance.lon, variance.lat, variance,
        transform=ccrs.PlateCarree(), cmap="viridis", vmin=0, vmax=vmax,
    )
    ax.coastlines(resolution="50m", linewidth=0.6)
    ax.set_extent([-180, 180, -80, 90], crs=ccrs.PlateCarree())
    # gridlines(draw_labels=True) computes a map-boundary polygon to place
    # labels along, which is degenerate for this extent in cartopy 0.25
    # (GEOSException: "Points of LinearRing do not form a closed linestring").
    # Plain set_xticks/set_yticks + cartopy's lon/lat formatters sidestep that
    # code path entirely -- this is the same pattern already used elsewhere
    # in this repo (2023a_SSH_mapping_OSE/src/mod_plot.py).
    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.5, linestyle="--")
    ax.set_xticks(range(-180, 181, 60), crs=ccrs.PlateCarree())
    ax.set_yticks(range(-60, 91, 30), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
    ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)

    # No domain-box overlays here -- kept only on the per-domain zoom panels
    # below, where they're the actual sharp-cornered rectangle deliverable.

    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02, shrink=0.8)
    cbar.set_label(cbar_label)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_domain_zoom(variance: xr.DataArray, leadtime: int, vmax: float, cbar_label: str, title: str) -> plt.Figure:
    fig, axs = plt.subplots(
        1, len(DOMAINS), figsize=(5 * len(DOMAINS), 5),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    for ax, (name, spec) in zip(axs, DOMAINS.items()):
        lon_min, lon_max = spec["lon"]
        lat_min, lat_max = spec["lat"]
        sub = variance.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
        mesh = ax.pcolormesh(
            sub.lon, sub.lat, sub,
            transform=ccrs.PlateCarree(), cmap="viridis", vmin=0, vmax=vmax,
        )
        ax.coastlines(resolution="50m", linewidth=0.6)
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        # sharp-cornered border around the whole panel too, for consistency
        rect = mpatches.Rectangle(
            (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
            transform=ccrs.PlateCarree(), fill=False,
            edgecolor=spec["color"], linewidth=2, zorder=5,
        )
        ax.add_patch(rect)
        ax.set_title(name, fontsize=11, fontweight="bold", color=spec["color"])

    cbar = fig.colorbar(mesh, ax=axs, orientation="vertical", pad=0.02, shrink=0.8)
    cbar.set_label(cbar_label)
    fig.suptitle(title)
    return fig


def main(leadtimes, season):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if season == "winter" else f"_{season}"
    for leadtime in leadtimes:
        variance, truth_variance = load_mean_variance(leadtime, DATA_DIRS[season])
        vmax = float(np.nanpercentile(variance, 99))
        raw_label = "Ensemble variance of SLA forecast [m$^2$]"
        raw_title = f"FM UNet (crps10) forecast variance -- leadtime {leadtime}d, 2023 {season} NRT mean"

        fig = plot_global_map(variance, leadtime, vmax, raw_label, raw_title)
        out_path = OUT_DIR / f"variance_map_fm_unet_crps10_leadtime{leadtime}{suffix}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", out_path)

        fig = plot_domain_zoom(variance, leadtime, vmax, raw_label,
                                f"FM UNet (crps10) forecast variance by domain -- leadtime {leadtime}d, 2023 {season} NRT mean")
        out_path = OUT_DIR / f"variance_domains_fm_unet_crps10_leadtime{leadtime}{suffix}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", out_path)

        # Normalized by the L3 truth field's own temporal variance -- a
        # scale-free view of how much of the ocean's natural SLA
        # variability the ensemble spread represents at each pixel.
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = variance / truth_variance
        ratio = ratio.where(np.isfinite(ratio))
        norm_label = "Ensemble variance / L3 truth temporal variance (unitless)"
        norm_vmax = float(np.nanpercentile(ratio, 99))
        norm_title = f"FM UNet (crps10) forecast variance, normalized by L3 truth variance -- leadtime {leadtime}d, 2023 {season}"

        fig = plot_global_map(ratio, leadtime, norm_vmax, norm_label, norm_title)
        out_path = OUT_DIR / f"variance_map_normalized_fm_unet_crps10_leadtime{leadtime}{suffix}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", out_path)

        fig = plot_domain_zoom(ratio, leadtime, norm_vmax, norm_label,
                                f"FM UNet (crps10) forecast variance / L3 truth variance, by domain -- leadtime {leadtime}d, 2023 {season}")
        out_path = OUT_DIR / f"variance_domains_normalized_fm_unet_crps10_leadtime{leadtime}{suffix}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--leadtimes", type=int, nargs="+", default=[0, 3, 5])
    parser.add_argument("--season", choices=sorted(DATA_DIRS), default="winter")
    args = parser.parse_args()
    main(args.leadtimes, args.season)
