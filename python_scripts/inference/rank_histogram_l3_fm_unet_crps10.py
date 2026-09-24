"""
Rank histogram (Talagrand diagram) of the valid-day L3 along-track SLA
within the FM UNet (crps10) ensemble, Gulf Stream box, over the full 90-day
winter member run saved by make_member_sla_animation_fm_unet_crps10.py.

For every (day, pixel) with an L3 observation on the forecast valid date,
the rank = number of members below the observation (0..N). A calibrated
ensemble gives a flat histogram; U-shaped = underdispersive (too little
spread); dome = overdispersive; sloped = biased (more mass at rank 0 means
observations below the members, i.e. the ensemble is too high).

At leadtime 0 the valid-day tracks are outside the model's 14-day
observation window, so they're independent of the forecast.

Two variants:
  - raw members
  - members + N(0, OBS_ERR^2) noise ("perturbed ensemble"), since L3
    unfiltered SLA has its own measurement noise, which alone would make a
    perfect ensemble look U-shaped.
Each for all track points and for high-spread points only (std > HIGH_STD).

Usage:
    python rank_histogram_l3_fm_unet_crps10.py [--leadtime 0]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

OBS_DAYS = 14
HIGH_STD = 0.10  # m
OBS_ERR = 0.03  # m, typical L3 unfiltered SLA noise
OUTPUTS = Path("/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/")
NPZ_DIR = OUTPUTS / "member_animation_crps10"
L3_PATH = "/Odyssey/public/altimetry_traces/nrt_sla/2023/gridded_input.nc"
OUT_DIR = OUTPUTS / "figures"


def main(leadtime):
    d = np.load(NPZ_DIR / f"member_fields_winter_leadtime{leadtime}.npz")
    fields, lat, lon = d["fields"], d["lat"], d["lon"]
    n_mem = fields.shape[1]
    valid = pd.to_datetime([str(x) for x in d["dates"]]) + pd.Timedelta(days=OBS_DAYS + leadtime)

    l3 = xr.open_dataset(L3_PATH)["sla_unfiltered"].sel(lat=lat, lon=lon, method="nearest")
    obs_all = l3.sel(time=valid).values  # (n_days, lat, lon)

    rng = np.random.default_rng(0)
    ranks = {k: [] for k in ["all", "high", "all_pert", "high_pert"]}
    bias = {"all": [], "high": []}
    for t in range(len(valid)):
        obs = obs_all[t]
        ok = np.isfinite(obs)
        if not ok.any():
            continue
        std = fields[t].std(0)
        mem = fields[t][:, ok]  # (n_mem, n_pts)
        pert = mem + rng.normal(0, OBS_ERR, mem.shape)
        o = obs[ok]
        hi = std[ok] > HIGH_STD
        r, rp = (mem < o).sum(0), (pert < o).sum(0)
        ranks["all"].append(r)
        ranks["all_pert"].append(rp)
        ranks["high"].append(r[hi])
        ranks["high_pert"].append(rp[hi])
        bias["all"].append(mem.mean(0) - o)
        bias["high"].append((mem.mean(0) - o)[hi])

    ranks = {k: np.concatenate(v) for k, v in ranks.items()}
    bias = {k: np.concatenate(v) for k, v in bias.items()}

    fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    expected = 1 / (n_mem + 1)
    titles = {"all": "All L3 points", "high": f"High-spread points (std > {HIGH_STD * 100:.0f} cm)"}
    for c, sub in enumerate(["all", "high"]):
        for r, (suffix, label) in enumerate([("", "raw members"), ("_pert", f"members + {OBS_ERR * 100:.0f} cm obs noise")]):
            k = sub + suffix
            freq = np.bincount(ranks[k], minlength=n_mem + 1) / ranks[k].size
            outside = freq[0] + freq[-1]
            ax = axs[r, c]
            ax.bar(np.arange(n_mem + 1), freq, color="#2a78d6")
            ax.axhline(expected, color="k", ls="--", lw=1, label="flat (calibrated)")
            ax.set_title(f"{titles[sub]}, {label}\nn={ranks[k].size}, outside ensemble {outside * 100:.0f}% "
                         f"(calibrated {2 * expected * 100:.0f}%)", fontsize=9)
            ax.set_ylabel("frequency")
            print(f"{k:10s} n={ranks[k].size:6d} outside={outside * 100:4.1f}%  "
                  f"rank0={freq[0] * 100:4.1f}%  rank{n_mem}={freq[-1] * 100:4.1f}%")
    for ax in axs[1]:
        ax.set_xlabel(f"rank of L3 observation among {n_mem} members")
    axs[0, 0].legend(frameon=False, fontsize=8)
    for sub in ["all", "high"]:
        print(f"mean(ensemble mean - L3) {sub}: {bias[sub].mean() * 100:+.1f} cm "
              f"(median {np.median(bias[sub]) * 100:+.1f} cm)")
    fig.suptitle(f"FM UNet (crps10) rank histogram vs valid-day L3 -- Gulf Stream, winter 2023, leadtime {leadtime}d\n"
                 f"mean bias (ens mean - L3): all {bias['all'].mean() * 100:+.1f} cm, "
                 f"high-spread {bias['high'].mean() * 100:+.1f} cm")
    fig.tight_layout()
    out = OUT_DIR / f"rank_histogram_l3_fm_unet_crps10_winter_leadtime{leadtime}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("Saved:", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--leadtime", type=int, default=0)
    main(parser.parse_args().leadtime)
