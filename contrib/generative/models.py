import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class GenFlowLit(pl.LightningModule):

    def __init__(
            self, 
            solver, 
            rec_weight, 
            opt_fn, 
            max_steps,
            rec_weight_fn=None, 
            norm_stats=None, 
            test_metrics=None, 
            pre_metric_fn=None, 
            persist_rw=True, 
            output_leadtime_start=None, 
            output_only_forecast=True
            ):
        super().__init__()
        self.register_buffer('rec_weight', torch.from_numpy(rec_weight), persistent=persist_rw)
        self.test_data = None
        self._norm_stats = norm_stats
        self.opt_fn = opt_fn
        self.metrics = test_metrics or {}
        self.pre_metric_fn = pre_metric_fn or (lambda x: x)

        self.rec_weight_fn = rec_weight_fn
        self.output_leadtime_start = output_leadtime_start
        self.output_only_forecast = output_only_forecast

        self.max_steps = max_steps
        self.solver = solver(max_steps=max_steps)
        
    # PYTORCH LIGHTNING LOGIC

    @property
    def norm_stats(self):
        if self._norm_stats is not None:
            return self._norm_stats
        elif self.trainer.datamodule is not None:
            return self.trainer.datamodule.norm_stats()
        return (0., 1.)

    @staticmethod
    def weighted_mse(err, weight):
        err_w = err * weight[None, ...]
        non_zeros = (torch.ones_like(err) * weight[None, ...]) == 0.0
        err_num = err.isfinite() & ~non_zeros
        if err_num.sum() == 0:
            return torch.scalar_tensor(1000.0, device=err_num.device).requires_grad_()
        loss = F.mse_loss(err_w[err_num], torch.zeros_like(err_w[err_num]))
        return loss
    
    @staticmethod
    def mask_batch(batch):

        # temporal masking
        new_input = batch.input
        dims = new_input.size()
        new_input[:, dims[1]//2:, :, :] = np.nan

        mask_batch = batch._replace(input=new_input)

        return mask_batch
    

    def gen_training_batch(self, batch):

        batch = self.mask_batch(batch)

        # gen algorithmic time
        batch_size = batch.input.size()[0]
        self.ts = torch.randint(self.max_steps, (batch_size,)).to(batch.input.device)

        # gen interpolated x_t
        self.x0s = torch.rand_like(batch.input)
        self.xts = self.x0s * (1 - self.ts.view(batch_size, 1, 1, 1) / self.max_steps) + (self.ts.view(batch_size, 1, 1, 1) / self.max_steps) * batch.tgt
        self.bs = batch.tgt - self.x0s

        return batch


    def forward(self, batch):
        return self.solver(xt=self.xts.nan_to_num(), y=batch.nan_to_num(), t=self.ts)

    def training_step(self, batch, batch_idx):
        batch = self.gen_training_batch(batch)
        return self.step(batch, "train")[0]

    def validation_step(self, batch, batch_idx):
        batch = self.gen_training_batch(batch)
        return self.step(batch, "val")[0]

    def step(self, batch, phase=""):
        if self.training and batch.tgt.isfinite().float().mean() < 0.1:
            return None, None

        out = self(batch=batch.input)
        loss = self.weighted_mse(out - self.bs, self.rec_weight)
        with torch.no_grad():
            self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss, out
    
    def configure_optimizers(self):
        return self.opt_fn(self)
    
    @property
    def test_quantities(self):
        return ['out']

    def clear_gpu_mem(self):
        del self.solver

        torch.cuda.empty_cache()

    def test_step(self, batch, batch_idx):
        pass

    def on_test_epoch_end(self):
        pass

    def sample(self, batch):

        batch_size = batch.input.size()[0]

        batch = self.mask_batch(batch)
        batch_input = batch.input.cuda()
        #batch_tgt = batch.tgt.cuda()

        #self.x0s = torch.rand_like(batch_input)
        #self.xts = self.x0s.clone()
        self.xts = torch.rand_like(batch.input)
        #self.bs = batch_tgt - self.x0s

        returns = []

        for t in tqdm(range(self.max_steps), position=1):
            self.ts = torch.ones((batch_size,)).type(torch.int).to(batch_input.device) * t
            out = self.solver(xt=self.xts.nan_to_num().to(batch_input.device), y=batch_input.nan_to_num(), t=self.ts.to(batch_input.device))

            self.xts += (out.detach().cpu() / self.max_steps)
            del out
            del self.ts
            torch.cuda.empty_cache()
            if t % 20 == 0:
                returns.append(self.xts.clone())

        returns.append(self.xts.clone())
        return returns

    def sample_sde(self, batch, epsilon):

        batch_size = batch.input.size()[0]

        batch = self.mask_batch(batch)
        batch_input = batch.input.cuda()
        #batch_tgt = batch.tgt.cuda()

        #self.x0s = torch.rand_like(batch_input)
        #self.xts = self.x0s.clone()
        self.xts = torch.rand_like(batch.input)
        #self.bs = batch_tgt - self.x0s

        returns = []

        for t in tqdm(range(self.max_steps)):
            self.ts = torch.ones((batch_size,)).type(torch.int).to(batch_input.device) * t
            out = self.solver(xt=self.xts.nan_to_num().to(batch_input.device), y=batch_input.nan_to_num(), t=self.ts.to(batch_input.device))

            Wt = torch.rand_like(batch.input) / self.max_steps
            eps_t = epsilon(t)

            t_1 = t / self.max_steps

            self.xts += self.xts * t_1 / self.max_steps + (out.detach().cpu() - t_1 * self.xts) * (t_1**2 - t_1 - 1 + eps_t) / (t_1**2 - t_1 -1) / self.max_steps + np.sqrt(2*eps_t) * Wt
            del out
            del self.ts
            if t % 20 == 0:
                returns.append(self.xts.clone())

        returns.append(self.xts.clone())
        return returns


def cosanneal_lr_adam(lit_mod, lr, T_max=100, weight_decay=0.):
    opt = torch.optim.Adam(
        [
            {"params": lit_mod.parameters(), "lr": lr},
        ], weight_decay=weight_decay
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max),
    }