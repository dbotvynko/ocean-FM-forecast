"""
Winter (Jan-Mar 2023) vs summer (Jul-Sep 2023) comparison of the FM UNet
(crps10) ensemble-variance maps of plot_variance_maps_fm_unet_crps10.py,
made so the two seasons can actually be compared:

  - ONE colour scale shared by both seasons (99th percentile of the pooled
    values), instead of each map's own 99th percentile (which made the
    seasons look more different than they are: ~6 vs ~9.5 for the
    normalized maps).
  - The normalized ratio's denominator (temporal variance of L3 SLA) is
    estimated from all L3 samples in a POOL x POOL pixel neighbourhood
    (5x5 = 1.25 deg) instead of the single pixel, which only has ~14 samples
    over 90 days in the open ocean (fewer between tracks, and a handful or
    none under seasonal sea ice). Pixels whose pooled sample count is below
    MIN_SAMPLES are masked in both raw and normalized maps. (A per-pixel
    ">= 8 samples" mask was tried first: it blanked ~100k open-ocean pixels
    between tracks and left the ice-edge hot spots in place.)

Separate files from the per-season originals (suffix "_seasons_masked"):
  - variance_map[_normalized]_fm_unet_crps10_leadtime{lt}_seasons_masked.png
    (global maps, winter above summer)
  - variance_domains[_normalized]_fm_unet_crps10_leadtime{lt}_seasons_masked.png
    (Gulfstream/Groenland/Madere, winter row above summer row)

Usage:
    python plot_variance_maps_seasons_fm_unet_crps10.py [--leadtimes 0 3 5] [--pool 5] [--min-samples 20]
"""

import argparse

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter

from plot_variance_maps_fm_unet_crps10 import DATA_DIRS, DOMAINS, OUT_DIR, load_mean_variance

SEASONS = ["winter", "summer"]
SEASON_LABELS = {"winter": "Jan-Mar 2023", "summer": "Jul-Sep 2023"}


def plot_global(maps, vmax, cbar_label, title):
    fig, axs = plt.subplots(2, 1, figsize=(12, 11), subplot_kw={"projection": ccrs.PlateCarree()})
    for ax, season in zip(axs, SEASONS):
        da = maps[season]
        mesh = ax.pcolormesh(da.lon, da.lat, da, transform=ccrs.PlateCarree(), cmap="viridis", vmin=0, vmax=vmax)
        ax.coastlines(resolution="50m", linewidth=0.6)
        ax.set_extent([-180, 180, -80, 90], crs=ccrs.PlateCarree())
        # Plain ticks, not gridlines(draw_labels=True): see plot_variance_maps_fm_unet_crps10.py
        ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.5, linestyle="--")
        ax.set_xticks(range(-180, 181, 60), crs=ccrs.PlateCarree())
        ax.set_yticks(range(-60, 91, 30), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
        ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
        ax.set_title(f"{season} ({SEASON_LABELS[season]}) -- median {float(da.median()):.2f}", fontsize=11)
    cbar = fig.colorbar(mesh, ax=axs, orientation="vertical", pad=0.02, shrink=0.7, extend="max")
    cbar.set_label(cbar_label)
    fig.suptitle(title)
    return fig


def plot_domains(maps, vmax, cbar_label, title):
    fig, axs = plt.subplots(2, len(DOMAINS), figsize=(5 * len(DOMAINS), 10),
                            subplot_kw={"projection": ccrs.PlateCarree()})
    for r, season in enumerate(SEASONS):
        for ax, (name, spec) in zip(axs[r], DOMAINS.items()):
            lon_min, lon_max = spec["lon"]
            lat_min, lat_max = spec["lat"]
            sub = maps[season].sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
            mesh = ax.pcolormesh(sub.lon, sub.lat, sub, transform=ccrs.PlateCarree(), cmap="viridis", vmin=0, vmax=vmax)
            ax.coastlines(resolution="50m", linewidth=0.6)
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            ax.add_patch(mpatches.Rectangle(
                (lon_min, lat_min), lon_max - lon_min, lat_max - lat_min, transform=ccrs.PlateCarree(),
                fill=False, edgecolor=spec["color"], linewidth=2, zorder=5,
            ))
            ax.set_title(f"{name} -- {season}, median {float(sub.median()):.2f}",
                         fontsize=11, fontweight="bold", color=spec["color"])
    cbar = fig.colorbar(mesh, ax=axs, orientation="vertical", pad=0.02, shrink=0.7, extend="max")
    cbar.set_label(cbar_label)
    fig.suptitle(title)
    return fig


def pooled_truth_variance(count, s1, s2, pool):
    """Temporal variance of L3 SLA from all samples in a pool x pool window
    (moments summed over the window; land contributes zero samples)."""
    def box_sum(a):
        return uniform_filter(np.nan_to_num(a.values), size=pool, mode="nearest") * pool ** 2
    n, t1, t2 = box_sum(count), box_sum(s1), box_sum(s2)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = t2 / n - (t1 / n) ** 2
    return count.copy(data=var), count.copy(data=n)


def main(leadtimes, pool, min_samples):
    for lt in leadtimes:
        raw, norm = {}, {}
        for season in SEASONS:
            variance, _, count, s1, s2 = load_mean_variance(lt, DATA_DIRS[season], return_counts=True)
            truth_variance, pooled_n = pooled_truth_variance(count, s1, s2, pool)
            enough = (pooled_n >= min_samples) & (count > 0)  # count > 0 keeps the land mask
            raw[season] = variance.where(enough)
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = variance / truth_variance
            norm[season] = ratio.where(enough & np.isfinite(ratio))
            print(f"  leadtime={lt} {season}: masked {int((~enough & np.isfinite(variance)).sum())} ocean pixels "
                  f"with < {min_samples} pooled L3 samples; normalized median {float(norm[season].median()):.3f}")

        mask_note = f"L3 variance pooled over {pool}x{pool} px, < {min_samples} samples masked"
        for maps, kind, label, name in [
            (raw, "raw", "Ensemble variance of SLA forecast [m$^2$]", "forecast variance"),
            (norm, "normalized", "Ensemble variance / L3 truth temporal variance (unitless)",
             "forecast variance / L3 truth variance"),
        ]:
            vmax = float(np.nanpercentile(np.concatenate([maps[s].values.ravel() for s in SEASONS]), 99))
            stem = "" if kind == "raw" else "_normalized"
            title = f"FM UNet (crps10) {name} -- leadtime {lt}d, winter vs summer (common scale, {mask_note})"

            fig = plot_global(maps, vmax, label, title)
            out = OUT_DIR / f"variance_map{stem}_fm_unet_crps10_leadtime{lt}_seasons_masked.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print("Saved:", out)

            fig = plot_domains(maps, vmax, label, title)
            out = OUT_DIR / f"variance_domains{stem}_fm_unet_crps10_leadtime{lt}_seasons_masked.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print("Saved:", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--leadtimes", type=int, nargs="+", default=[0, 3, 5])
    parser.add_argument("--pool", type=int, default=5)
    parser.add_argument("--min-samples", type=int, default=20)
    args = parser.parse_args()
    main(args.leadtimes, args.pool, args.min_samples)
