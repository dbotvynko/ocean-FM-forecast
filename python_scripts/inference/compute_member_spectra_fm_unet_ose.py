"""
Per-member Gulf Stream wavenumber spectra for any FM UNet xp/checkpoint (used
for the OSE run), same diagnostic as compute_member_spectra_fm_unet_crps10.py
(left untouched): reruns the 10-member inference over the 90 days of a season,
keeps each member's isotropic PSD in the Gulf Stream box, discards the fields.

The inference loop, box, PSD estimator, GLO12/deterministic matching and the
per-member figure are imported from that script. Adds a comparison figure
of these members against the OSSE (crps10) members already saved in
outputs/member_spectra_crps10/member_spectra_leadtime<lt>[_summer].npz.

Default input is sla_unfiltered, as in the OSSE member spectra, so the two
are directly comparable.

GPU, same cost as the eval: ~10 h for 90 days x 10 members on one L40s.

Usage:
    python compute_member_spectra_fm_unet_ose.py --xp XP --ckpt CKPT --tag TAG
        [--leadtimes 0 3 5] [--season winter|summer] [--var sla_unfiltered]
"""

import argparse
import sys
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from contrib.generative.inference import (  # noqa: E402
    YearlyLeadtimeEvaluator, coord_builder_for, load_gen_flow_checkpoint, load_gridded_sla,
)
from compute_member_spectra_fm_unet_crps10 import (  # noqa: E402
    NPZ_DIR as OSSE_NPZ_DIR, NRT_2023_PATH, NUM_SAMPLES, OUT_DIR, SEASON_RANGES,
    plot_member_spectra, run_all_leadtimes,
)

OUTPUTS = Path("/Odyssey/private/d21botvy/forecast/ocean-DDPMs/outputs/")


def plot_vs_osse(leadtime, r, osse, tag, season):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, psd_members, color in ((tag, r["psd_members"], "#1b9e77"),
                                     ("OSSE", osse["psd_members"], "#2a78d6")):
        per_member = np.nanmean(psd_members, axis=0)  # (members, nfreq), averaged over days
        ax.fill_between(r["freq_r"], per_member.min(0), per_member.max(0), color=color, alpha=0.2)
        ax.plot(r["freq_r"], np.median(per_member, axis=0), color=color, linewidth=2,
                label=f"FM UNet {name}: members (median, min-max), {psd_members.shape[0]} days")
    ax.plot(r["freq_r"], r["psd_det"], color="#eb6834", linewidth=2, linestyle="--", label="Deterministic UNet")
    if r["psd_glo12"] is not None:
        ax.plot(r["freq_r"], r["psd_glo12"], color="#2a2a2a", linewidth=2, linestyle=":",
                label="GLO12 (CMEMS forecast, regridded)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber [cycles/km]")
    ax.set_ylabel("PSD [m$^2$/(cycles/km)]")
    ax.set_title(f"Gulf Stream box spectra per member -- leadtime {leadtime}d\n"
                 f"FM UNet {tag} vs OSSE members, 2023 {season} NRT")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.3)
    fig.tight_layout()
    return fig


def main(args):
    npz_dir = OUTPUTS / f"member_spectra_{args.tag}"
    npz_dir.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "config")):
        cfg = compose(config_name="main", overrides=[f"xp={args.xp}"])
    model = load_gen_flow_checkpoint(hydra.utils.instantiate(cfg.model), args.ckpt)
    print("xp:", args.xp, "| ckpt:", args.ckpt, "| input:", args.var, flush=True)

    norm_stats = tuple(cfg.datamodule.norm_stats.train)
    patch_time = cfg.datamodule.xrds_kw.train.patch_dims.time
    domain_train = hydra.utils.instantiate(cfg.domain.train)
    sla_da = load_gridded_sla(NRT_2023_PATH, var=args.var,
                              lat_slice=domain_train["lat"], lon_slice=domain_train["lon"])

    season_start, season_end = SEASON_RANGES[args.season]
    start_dates = pd.date_range(season_start, season_end, freq="D")
    evaluator = YearlyLeadtimeEvaluator(
        model, sla_da, norm_stats, patch_time=patch_time, leadtimes=args.leadtimes,
        num_samples=NUM_SAMPLES, coord_builder=coord_builder_for(cfg),
    )
    results = run_all_leadtimes(evaluator, patch_time // 2, start_dates, args.leadtimes)

    suffix = "" if args.season == "winter" else f"_{args.season}"
    for lt in args.leadtimes:
        r = results[lt]
        np.savez(npz_dir / f"member_spectra_leadtime{lt}{suffix}.npz", freq_r=r["freq_r"],
                 psd_members=r["psd_members"], psd_det=r["psd_det"],
                 psd_glo12=r["psd_glo12"] if r["psd_glo12"] is not None else np.array([]),
                 n_matched=r["n_matched"], n_matched_glo12=r["n_matched_glo12"])

        fig = plot_member_spectra(lt, r["freq_r"], r["psd_members"], r["psd_det"], r["n_matched"],
                                  r["psd_glo12"], r["n_matched_glo12"], args.season)
        out = OUT_DIR / f"member_spectra_fm_unet_{args.tag}_leadtime{lt}{suffix}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", out, flush=True)

        osse_npz = OSSE_NPZ_DIR / f"member_spectra_leadtime{lt}{suffix}.npz"
        if osse_npz.exists():
            fig = plot_vs_osse(lt, r, np.load(osse_npz), args.tag, args.season)
            out = OUT_DIR / f"member_spectra_fm_unet_{args.tag}_vs_osse_leadtime{lt}{suffix}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print("Saved:", out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xp", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--leadtimes", type=int, nargs="+", default=[0, 3, 5])
    parser.add_argument("--season", choices=sorted(SEASON_RANGES), default="winter")
    parser.add_argument("--var", default="sla_unfiltered")
    main(parser.parse_args())
