"""
Animated (MP4) film of per-member SLA *anomalies* (member minus ensemble
mean) over the 90-day winter (Jan-Mar 2023) window, Gulf Stream box,
leadtime 0, for the base FM UNet (crps10) ensemble.

Companion to make_member_sla_animation_fm_unet_crps10.py: there the raw
member fields share one colour scale set by the full SLA amplitude
(~ +/-50 cm), so the few-cm member-to-member differences are invisible.
Here the ensemble mean is removed, so the anomaly panels get their own,
much tighter colour scale and the spread becomes visible.

No GPU needed: rebuilt purely from the .npz saved by the raw-field script.

Layout (3 rows x 4 cols):
  - column 0: ensemble mean SLA, ensemble std, box-mean spread vs time
  - columns 1-3: anomalies of members 0-8 (3x3)
The ensemble mean / std use all 10 members; only 9 anomalies are plotted.

Usage:
    python make_member_anomaly_animation_fm_unet_crps10.py [--leadtime 0]
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LEADTIME = 0  # overridden by --leadtime
NUM_MEMBERS_TO_PLOT = 9
# The .npz "dates" are the start dates of each day's 14-day observation
# window; the forecast shown is valid obs_days + leadtime days later
# (same convention as YearlyLeadtimeEvaluator / leadtime_indices).
OBS_DAYS = 14
NPZ_DIR = Path("/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/member_animation_crps10/")
OUT_DIR = Path("/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/figures/")


def build_animation(fields, lat, lon, dates, out_path):
    ens_mean = np.nanmean(fields, axis=1)  # (n_days, lat, lon)
    ens_std = np.nanstd(fields, axis=1)
    anoms = fields[:, :NUM_MEMBERS_TO_PLOT] - ens_mean[:, None]
    box_spread = np.nanmean(ens_std, axis=(1, 2))
    init_dates = pd.to_datetime([str(d) for d in dates])
    valid_dates = init_dates + pd.Timedelta(days=OBS_DAYS + LEADTIME)

    # Robust, symmetric scales computed over the whole run so frames are comparable
    mean_abs = np.nanpercentile(np.abs(ens_mean), 99)
    anom_abs = np.nanpercentile(np.abs(anoms), 99)
    std_max = np.nanpercentile(ens_std, 99)

    # Extra horizontal room (set before any colorbar steals space): column 0
    # carries its own colorbars, whose labels otherwise run into the y tick
    # labels of column 1.
    fig, axs = plt.subplots(3, 4, figsize=(15.5, 10.5), sharex=False, sharey=False, gridspec_kw={"wspace": 0.45})
    kw = dict(shading="auto")

    mean_mesh = axs[0, 0].pcolormesh(lon, lat, ens_mean[0], cmap="RdBu_r", vmin=-mean_abs, vmax=mean_abs, **kw)
    axs[0, 0].set_title("Ensemble mean SLA", fontsize=10)
    fig.colorbar(mean_mesh, ax=axs[0, 0], label="SLA [m]", shrink=0.85)

    std_mesh = axs[1, 0].pcolormesh(lon, lat, ens_std[0], cmap="viridis", vmin=0, vmax=std_max, **kw)
    axs[1, 0].set_title("Ensemble std (10 members)", fontsize=10)
    fig.colorbar(std_mesh, ax=axs[1, 0], label="std [m]", shrink=0.85)

    ax_ts = axs[2, 0]
    ax_ts.plot(np.arange(len(dates)), box_spread * 100, color="0.3", lw=1.2)
    marker, = ax_ts.plot([0], [box_spread[0] * 100], "o", color="C3")
    ax_ts.set_title("Box-mean ensemble std", fontsize=10)
    ax_ts.set_xlabel(f"day (valid {valid_dates[0].date()} to {valid_dates[-1].date()})")
    ax_ts.set_ylabel("std [cm]")
    ax_ts.grid(alpha=0.3)

    anom_meshes = []
    for m in range(NUM_MEMBERS_TO_PLOT):
        ax = axs[m // 3, 1 + m % 3]
        mesh = ax.pcolormesh(lon, lat, anoms[0, m], cmap="PuOr_r", vmin=-anom_abs, vmax=anom_abs, **kw)
        ax.set_title(f"Member {m} - mean", fontsize=10)
        anom_meshes.append(mesh)
    fig.colorbar(anom_meshes[0], ax=axs[:, 1:], orientation="vertical", shrink=0.8, label="SLA anomaly [m]")

    for ax in axs[:2, 0].tolist() + [axs[r, c] for r in range(3) for c in range(1, 4)]:
        ax.set_aspect("equal")
        ax.tick_params(labelsize=8)

    suptitle = fig.suptitle("", fontsize=13)

    def update(frame):
        mean_mesh.set_array(ens_mean[frame].ravel())
        std_mesh.set_array(ens_std[frame].ravel())
        for m, mesh in enumerate(anom_meshes):
            mesh.set_array(anoms[frame, m].ravel())
        marker.set_data([frame], [box_spread[frame] * 100])
        suptitle.set_text(
            f"FM UNet (crps10) member anomalies (member - ensemble mean) -- Gulf Stream, leadtime {LEADTIME}d\n"
            f"valid {valid_dates[frame].date()} (init {init_dates[frame].date()}) ({frame + 1}/{len(dates)})"
        )
        return [mean_mesh, std_mesh, marker, suptitle] + anom_meshes

    anim = animation.FuncAnimation(fig, update, frames=len(dates), blit=False)
    anim.save(out_path, writer="ffmpeg", fps=1, dpi=120)
    plt.close(fig)
    print(f"Scales: mean +/-{mean_abs:.3f} m, anomaly +/-{anom_abs:.3f} m, std 0-{std_max:.3f} m")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = np.load(NPZ_DIR / f"member_fields_winter_leadtime{LEADTIME}.npz")
    out_path = OUT_DIR / f"member_anomaly_animation_fm_unet_crps10_winter_leadtime{LEADTIME}.mp4"
    build_animation(d["fields"], d["lat"], d["lon"], d["dates"], out_path)
    print("Saved animation:", out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--leadtime", type=int, default=LEADTIME)
    LEADTIME = parser.parse_args().leadtime
    main()
