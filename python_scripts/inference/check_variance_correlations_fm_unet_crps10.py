"""
Check what the FM UNet (crps10) ensemble-variance maps actually correlate
with, for the 2023 NRT evaluation:

  1. Spread-skill check: does forecast_std**2 (ensemble variance) spatially
     correlate with crps_fair (a proper, ensemble-size-corrected local error
     score, already saved per-pixel in the eval .nc files)? A well-calibrated
     ensemble should show high spread where local error is also high.

  2. EKE-proxy confound check: does the variance correlate with mesoscale
     activity itself, rather than (or in addition to) forecast error? We
     don't have a true eddy-kinetic-energy product on this grid, so we use
     the temporal variance of the "truth" SLA field across the 2023 record
     as a standard proxy for mesoscale variability/EKE (regions of strong
     currents like the Gulf Stream show high SLA variance).

Both forecast_std and crps_fair are defined by the model everywhere,
including land, so they're masked using "truth" SLA's own NaN pattern
(land-masked by construction) before plotting or computing correlations.

For each leadtime, saves 4 separate figures:
  - spread_skill_maps_fm_unet_crps10_leadtime{L}.png   (variance | crps_fair maps)
  - spread_skill_scatter_fm_unet_crps10_leadtime{L}.png (scatter + Pearson r)
  - spread_eke_proxy_maps_fm_unet_crps10_leadtime{L}.png   (variance | SLA-variance maps)
  - spread_eke_proxy_scatter_fm_unet_crps10_leadtime{L}.png (scatter + Pearson r)

Usage:
    python check_variance_correlations_fm_unet_crps10.py [--leadtimes 0 3 5] [--season winter|summer]
"""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Must match eval_nrt_2023_fm_unet_crps10.py's --season output dirs.
SEASON_DATA_DIRS = {
    "winter": Path("/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/eval_nrt2023_fm_unet_crps10/"),
    "summer": Path("/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/eval_nrt2023_fm_unet_crps10_summer/"),
}
OUT_DIR = Path("/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/figures/")


def load_fields(leadtime: int, data_dir: Path):
    """One pass over all daily files, accumulating everything we need:
    mean forecast variance, mean crps_fair (spread-skill), and the temporal
    variance of the truth SLA field (EKE proxy)."""
    files = sorted(data_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No .nc files found in {data_dir}")

    sum_var = sum_crps = sum_truth = sum_truth2 = None
    count_var = count_crps = count_truth = None
    lat = lon = None
    skipped = []

    for f in files:
        with xr.open_dataset(f) as ds:
            if "leadtime" not in ds.sizes or leadtime not in range(ds.sizes["leadtime"]):
                skipped.append(f.name)
                continue

            std = ds["forecast_std"].isel(leadtime=leadtime).values
            crps_fair = ds["crps_fair"].isel(leadtime=leadtime).values
            truth = ds["truth"].isel(leadtime=leadtime).values

            if lat is None:
                lat = ds["lat"].values
                lon = ds["lon"].values
                shape = std.shape
                sum_var, count_var = np.zeros(shape), np.zeros(shape)
                sum_crps, count_crps = np.zeros(shape), np.zeros(shape)
                sum_truth, sum_truth2, count_truth = np.zeros(shape), np.zeros(shape), np.zeros(shape)

            v = np.isfinite(std)
            sum_var[v] += std[v].astype(np.float64) ** 2
            count_var[v] += 1

            v = np.isfinite(crps_fair)
            sum_crps[v] += crps_fair[v].astype(np.float64)
            count_crps[v] += 1

            v = np.isfinite(truth)
            sum_truth[v] += truth[v].astype(np.float64)
            sum_truth2[v] += truth[v].astype(np.float64) ** 2
            count_truth[v] += 1

    if skipped:
        print(f"  (leadtime={leadtime}: skipped {len(skipped)}/{len(files)} files without it, "
              f"e.g. {skipped[:3]})")
    if lat is None:
        raise RuntimeError(f"No file had a valid leadtime={leadtime} slice")

    with np.errstate(invalid="ignore"):
        mean_variance = np.where(count_var > 0, sum_var / count_var, np.nan)
        mean_crps_fair = np.where(count_crps > 0, sum_crps / count_crps, np.nan)
        mean_truth = np.where(count_truth > 0, sum_truth / count_truth, np.nan)
        mean_truth2 = np.where(count_truth > 0, sum_truth2 / count_truth, np.nan)
        truth_variance = mean_truth2 - mean_truth ** 2

    # Land mask: forecast_std/crps_fair are defined by the model everywhere
    # including land, but "truth" SLA is properly NaN there -- reuse that as
    # the ocean mask so land pixels don't show up as spurious variance/error.
    land = count_truth == 0
    mean_variance = np.where(land, np.nan, mean_variance)
    mean_crps_fair = np.where(land, np.nan, mean_crps_fair)

    coords = {"lat": lat, "lon": lon}
    dims = ["lat", "lon"]
    return (
        xr.DataArray(mean_variance, coords=coords, dims=dims),
        xr.DataArray(mean_crps_fair, coords=coords, dims=dims),
        xr.DataArray(truth_variance, coords=coords, dims=dims),
    )


def pearson_r(x: xr.DataArray, y: xr.DataArray) -> float:
    xf, yf = x.values.ravel(), y.values.ravel()
    valid = np.isfinite(xf) & np.isfinite(yf)
    return float(np.corrcoef(xf[valid], yf[valid])[0, 1])


def _setup_map_axis(ax):
    ax.coastlines(resolution="50m", linewidth=0.6)
    ax.set_extent([-180, 180, -80, 90], crs=ccrs.PlateCarree())
    # See plot_variance_maps_fm_unet_crps10.py: gridlines(draw_labels=True)
    # crashes here (degenerate map-boundary polygon in cartopy 0.25), so use
    # plain ticks + formatters instead.
    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.5, linestyle="--")
    ax.set_xticks(range(-180, 181, 60), crs=ccrs.PlateCarree())
    ax.set_yticks(range(-60, 91, 30), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
    ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)


def plot_two_maps(field_a, field_b, label_a, label_b, cmap_a, cmap_b, title) -> plt.Figure:
    fig, axs = plt.subplots(
        1, 2, figsize=(15, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    for ax, field, label, cmap in [(axs[0], field_a, label_a, cmap_a), (axs[1], field_b, label_b, cmap_b)]:
        vmax = float(np.nanpercentile(field, 99))
        mesh = ax.pcolormesh(
            field.lon, field.lat, field,
            transform=ccrs.PlateCarree(), cmap=cmap, vmin=0, vmax=vmax,
        )
        _setup_map_axis(ax)
        ax.set_title(label, fontsize=11)
        cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02, shrink=0.75)
        cbar.set_label(label)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_scatter(x, y, xlabel, ylabel, r, title) -> plt.Figure:
    xf, yf = x.values.ravel(), y.values.ravel()
    valid = np.isfinite(xf) & np.isfinite(yf)
    xf, yf = xf[valid], yf[valid]

    # Subsample for a readable scatter -- ~65000 points (global 0.25deg-ish
    # grid) render as an unreadable solid blob otherwise.
    rng = np.random.default_rng(0)
    if xf.size > 20000:
        idx = rng.choice(xf.size, 20000, replace=False)
        xf, yf = xf[idx], yf[idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xf, yf, s=3, alpha=0.15, edgecolor="none", color="#2a78d6")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nPearson r = {r:.3f}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def main(leadtimes, data_dir, season):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if season == "winter" else f"_{season}"
    for leadtime in leadtimes:
        variance, crps_fair, truth_variance = load_fields(leadtime, data_dir)

        # --- Point 2: spread-skill check ---
        r_skill = pearson_r(variance, crps_fair)
        fig = plot_two_maps(
            variance, crps_fair,
            "Ensemble variance [m$^2$]", "Fair CRPS (local error) [m]",
            "viridis", "Reds",
            f"Spread vs. skill -- FM UNet (crps10), leadtime {leadtime}d, 2023 {season} NRT mean (r={r_skill:.3f})",
        )
        out_path = OUT_DIR / f"spread_skill_maps_fm_unet_crps10_leadtime{leadtime}{suffix}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", out_path, f"  (Pearson r = {r_skill:.3f})")

        fig = plot_scatter(
            variance, crps_fair,
            "Ensemble variance [m$^2$]", "Fair CRPS [m]", r_skill,
            f"Spread vs. skill, leadtime {leadtime}d ({season})",
        )
        out_path = OUT_DIR / f"spread_skill_scatter_fm_unet_crps10_leadtime{leadtime}{suffix}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", out_path)

        # --- Point 3: EKE-proxy confound check ---
        r_eke = pearson_r(variance, truth_variance)
        fig = plot_two_maps(
            variance, truth_variance,
            "Ensemble variance [m$^2$]", "Truth SLA variance (EKE proxy) [m$^2$]",
            "viridis", "viridis",
            f"Spread vs. EKE proxy -- FM UNet (crps10), leadtime {leadtime}d, 2023 {season} NRT mean (r={r_eke:.3f})",
        )
        out_path = OUT_DIR / f"spread_eke_proxy_maps_fm_unet_crps10_leadtime{leadtime}{suffix}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", out_path, f"  (Pearson r = {r_eke:.3f})")

        fig = plot_scatter(
            variance, truth_variance,
            "Ensemble variance [m$^2$]", "Truth SLA variance [m$^2$]", r_eke,
            f"Spread vs. EKE proxy, leadtime {leadtime}d ({season})",
        )
        out_path = OUT_DIR / f"spread_eke_proxy_scatter_fm_unet_crps10_leadtime{leadtime}{suffix}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--leadtimes", type=int, nargs="+", default=[0, 3, 5])
    parser.add_argument("--season", choices=sorted(SEASON_DATA_DIRS), default="winter")
    args = parser.parse_args()
    main(args.leadtimes, SEASON_DATA_DIRS[args.season], args.season)
