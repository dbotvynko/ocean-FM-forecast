"""
Training entrypoints complementing ocean4dvarnet.train.base_training.
"""

import torch


def warm_start_training(trainer, dm, lit_mod, init_ckpt):
    """
    Train `lit_mod` starting from the weights of `init_ckpt`, with a fresh
    optimizer, LR schedule and epoch counter.

    Unlike base_training(ckpt=...), which resumes the full training state
    (Adam moments, scheduler position, epoch), this only reuses the network
    weights -- e.g. to continue a run that diverged, from its best checkpoint,
    with a lower learning rate.
    """
    state = torch.load(init_ckpt, map_location="cpu", weights_only=False)["state_dict"]
    lit_mod.load_state_dict(state, strict=True)
    print(f"Warm start from {init_ckpt}")

    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    trainer.fit(lit_mod, datamodule=dm)
    trainer.test(lit_mod, datamodule=dm, ckpt_path="best")
