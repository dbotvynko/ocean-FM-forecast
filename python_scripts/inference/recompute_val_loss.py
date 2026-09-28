"""
Recompute the flow-matching val_loss of several checkpoints under identical
conditions, so runs whose logged val_loss isn't comparable can be ranked:

  - the base / plain-coords xps logged val_loss without sync_dist, i.e. only
    rank 0's half of the val set under 2-GPU DDP, while the Fourier xps
    averaged over both ranks;
  - the FM loss is stochastic (random t and x0 per batch).

Here every checkpoint sees the full val set on one GPU, in fp32, with the
same per-window seed (so the same t / x0 draws for every model), repeated
over NUM_SEEDS seeds. Reported: mean over windows, plus the spread across
seeds, and the same mean restricted to even / odd windows (roughly what
each DDP rank saw) to show how much the half-set logging alone moves it.

Usage: python recompute_val_loss.py
"""

import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from contrib.generative.inference import load_gen_flow_checkpoint  # noqa: E402

OUT = REPO_ROOT / "outputs"
RUNS = {
    "no_coords": ("forecast_DDPM_UNet_1patch",
                  "2026-09-01/13-42-20/forecast_DDPM_UNet_1patch/checkpoints/val_loss=0.01161-epoch=153.ckpt"),
    "coords": ("forecast_DDPM_UNet_1patch_coords",
               "2026-09-22/21-11-34/forecast_DDPM_UNet_1patch_coords/checkpoints/val_loss=0.01633-epoch=090.ckpt"),
    "coords_warmrestart": ("forecast_DDPM_UNet_1patch_coords_warmrestart",
                           "2026-09-24/13-23-15/forecast_DDPM_UNet_1patch_coords_warmrestart/checkpoints/val_loss=0.01527-epoch=129.ckpt"),
    "fourier": ("forecast_DDPM_UNet_1patch_coords_fourier",
                "2026-09-24/15-24-07/forecast_DDPM_UNet_1patch_coords_fourier/checkpoints/val_loss=0.01517-epoch=298.ckpt"),
    "learned_fourier": ("forecast_DDPM_UNet_1patch_coords_learned_fourier",
                        "2026-09-24/17-12-53/forecast_DDPM_UNet_1patch_coords_learned_fourier/checkpoints/val_loss=0.01413-epoch=247.ckpt"),
}
NUM_SEEDS = 3


@torch.no_grad()
def window_losses(model, val_dl, seed):
    losses = []
    for i, batch in enumerate(val_dl):
        batch = batch._replace(**{k: v.cuda() for k, v in batch._asdict().items()})
        torch.manual_seed(seed * 100_000 + i)
        batch = model.gen_training_batch(batch)
        out = model(batch=batch) if hasattr(batch, "coords") else model(batch=batch.input)
        losses.append(model.weighted_mse(out - model.bs, model.rec_weight).item())
    return np.array(losses)


results = {}
for name, (xp, ckpt) in RUNS.items():
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "config")):
        cfg = compose(config_name="main", overrides=[f"xp={xp}"])
    model = load_gen_flow_checkpoint(hydra.utils.instantiate(cfg.model), OUT / ckpt)
    model.rec_weight = model.rec_weight.cuda() if torch.is_tensor(model.rec_weight) else model.rec_weight
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.setup()
    val_dl = torch.utils.data.DataLoader(dm.val_ds, batch_size=1, shuffle=False, num_workers=8)

    per_seed = np.stack([window_losses(model, val_dl, s) for s in range(NUM_SEEDS)])  # (seeds, windows)
    results[name] = per_seed
    np.save(OUT / f"val_loss_recomputed_{name}.npy", per_seed)
    m = per_seed.mean(axis=1)
    print(f"{name:20s} n_windows={per_seed.shape[1]}  val_loss={m.mean():.5f} "
          f"(seed spread {m.min():.5f}-{m.max():.5f})  "
          f"even-half={per_seed[:, 0::2].mean():.5f} odd-half={per_seed[:, 1::2].mean():.5f}", flush=True)
    del model
    torch.cuda.empty_cache()

base = results["no_coords"].mean(axis=0)
print("\nPaired per-window difference vs no_coords (same windows, same t/x0 draws):")
for name, per_seed in results.items():
    if name == "no_coords":
        continue
    d = per_seed.mean(axis=0) - base
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"{name:20s} mean diff={d.mean():+.5f} +- {1.96 * se:.5f} (95%)  "
          f"windows where better than no_coords: {(d < 0).mean():.0%}")
