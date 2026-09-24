"""
On the highest-spread days of the winter leadtime-0 Gulf Stream member run
(make_member_sla_animation_fm_unet_crps10.py), check which ensemble
"scenario" the observations support.

References, both on the forecast *valid* date:
  - L3 along-track SLA (sla_unfiltered from the NRT gridded_input.nc, i.e.
    the model's own input product). At leadtime 0 the model only saw the 14
    days *before* the valid date, so these tracks are independent -- but
    sparse (a few-20% of the box per day).
  - DUACS DT L4 (cmems ...my_allsat-l4-duacs, 0.125 deg, box-regridded to
    the 1/4 deg model grid). Full map, but delayed-time: its optimal
    interpolation uses observations before *and after* the date, so it is a
    best-estimate reference, not an independent one.

Per day: per-member RMSE vs each reference, over the whole box and over the
high-spread pixels only (ensemble std > HIGH_STD), the ensemble-mean RMSE,
and the rank of the L3 values within the ensemble at high-spread track
points. Plus one figure per day: ensemble mean / std / DUACS / L3 tracks and
the best- and worst-matching members.

Usage:
    python check_peak_spread_scenarios_fm_unet_crps10.py [--n-days 3]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

LEADTIME = 0
OBS_DAYS = 14
HIGH_STD = 0.10  # m
OUTPUTS = Path("/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/")
NPZ_PATH = OUTPUTS / f"member_animation_crps10/member_fields_winter_leadtime{LEADTIME}.npz"
L3_PATH = "/Odyssey/public/altimetry_traces/nrt_sla/2023/gridded_input.nc"
DUACS_PATH = "/Odyssey/public/duacs/2023/duacs_2023_sla_adt.nc"
OUT_DIR = OUTPUTS / "figures"


def rmse(a, b, mask):
    m = mask & np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else np.nan, int(m.sum())


def main(n_days):
    d = np.load(NPZ_PATH)
    fields, lat, lon = d["fields"], d["lat"], d["lon"]
    valid = pd.to_datetime([str(x) for x in d["dates"]]) + pd.Timedelta(days=OBS_DAYS + LEADTIME)
    ens_mean, ens_std = fields.mean(1), fields.std(1)
    top = np.argsort(ens_std.mean((1, 2)))[::-1][:n_days]

    l3 = xr.open_dataset(L3_PATH)["sla_unfiltered"]
    duacs = xr.open_dataset(DUACS_PATH)["sla"]

    for t in top:
        vd = valid[t]
        obs = l3.sel(time=vd, method="nearest").sel(lat=lat, lon=lon, method="nearest").values
        ref = (duacs.sel(time=vd, method="nearest")
               .interp(latitude=lat, longitude=lon).values.astype(np.float32))
        high = ens_std[t] > HIGH_STD
        box = np.ones_like(high)

        print(f"\n=== valid {vd.date()} (frame {t + 1}), box-mean std {ens_std[t].mean() * 100:.1f} cm, "
              f"high-spread pixels {int(high.sum())} ===")
        rows = []
        for name, x in [*((f"member {m}", fields[t, m]) for m in range(fields.shape[1])), ("ENS MEAN", ens_mean[t])]:
            rows.append(dict(
                name=name,
                duacs_box=rmse(x, ref, box)[0], duacs_high=rmse(x, ref, high)[0],
                l3_box=rmse(x, obs, box)[0], l3_high=rmse(x, obs, high)[0],
            ))
        df = pd.DataFrame(rows).set_index("name") * 100
        print(f"RMSE [cm]  (L3 points: box {rmse(obs, obs, box)[1]}, high-spread {rmse(obs, obs, high)[1]})")
        print(df.round(1).to_string())

        # Rank of the observation within the 10 members at high-spread track points
        pts = high & np.isfinite(obs)
        if pts.any():
            ranks = (fields[t][:, pts] < obs[pts]).sum(0)
            print("L3 rank within ensemble at high-spread points (0 = below all members, 10 = above all):",
                  np.bincount(ranks, minlength=11).tolist())

        members_only = df.drop("ENS MEAN")
        best = int(members_only["duacs_high"].idxmin().split()[1])
        worst = int(members_only["duacs_high"].idxmax().split()[1])

        fig, axs = plt.subplots(2, 3, figsize=(15, 9.5), gridspec_kw={"wspace": 0.3})
        kw = dict(cmap="RdBu_r", vmin=-0.7, vmax=0.7, shading="auto")
        panels = [
            (ens_mean[t], "Ensemble mean"),
            (ref, "DUACS DT L4 (regridded)"),
            (fields[t, best], f"Best member vs DUACS (high-spread): {best}"),
            (None, "Ensemble std + L3 tracks"),
            (obs, "L3 along-track SLA (valid day, unseen)"),
            (fields[t, worst], f"Worst member vs DUACS (high-spread): {worst}"),
        ]
        for ax, (x, title) in zip(axs.ravel(), panels):
            if x is None:
                mesh = ax.pcolormesh(lon, lat, ens_std[t], cmap="viridis", vmin=0, vmax=0.25, shading="auto")
                fig.colorbar(mesh, ax=ax, label="std [m]", shrink=0.8, orientation="horizontal", pad=0.12)
                ax.contour(lon, lat, high, levels=[0.5], colors="w", linewidths=0.8)
                ty, tx = np.nonzero(np.isfinite(obs))
                ax.scatter(lon[tx], lat[ty], s=2, c="k")
            else:
                mesh = ax.pcolormesh(lon, lat, x, **kw)
                ax.contour(lon, lat, high, levels=[0.5], colors="k", linewidths=0.6)
            ax.set_title(title, fontsize=10)
            ax.set_aspect("equal")
        fig.colorbar(mesh, ax=axs, label="SLA [m]", shrink=0.6, location="right")
        fig.suptitle(f"FM UNet (crps10) Gulf Stream, leadtime {LEADTIME}d -- valid {vd.date()} "
                     f"(contour: ensemble std > {HIGH_STD * 100:.0f} cm)")
        out = OUT_DIR / f"peak_spread_check_fm_unet_crps10_leadtime{LEADTIME}_{vd.date()}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-days", type=int, default=3)
    main(parser.parse_args().n_days)
