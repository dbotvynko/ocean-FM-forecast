"""
Learning GLORYS12 data
"""

import functools as ft
import time

import numpy as np
import torch
import kornia.filters as kfilts
import xarray as xr
import pytorch_lightning as pl

from collections import namedtuple
from ocean4dvarnet.data import BaseDataModule, TrainingItem
from ocean4dvarnet.models import Lit4dVarNet

from omegaconf import OmegaConf
from pathlib import Path
import hydra

# Exceptions
# ----------

TrainingItemwithLonLat = namedtuple('TrainingItemwithLonLat', ['input', 'tgt','lon','lat'])
TrainingItemOSEwOSSE = namedtuple('TrainingItemOSEwOSSE', ['input', 'tgt','input_osse','tgt_osse','lon','lat'])
TrainingItemOSEwOSSEwMask = namedtuple('TrainingItemOSEwOSSEwMask', ['input', 'tgt','input_osse','tgt_osse','lon','lat','mask_input_lr'])

class NormParamsNotProvided(Exception):
    """Normalisation parameters have not been provided"""


# Data
# ----


class DistinctNormDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input_mask = None

        if isinstance(self.input_da, (tuple, list)):
            self.input_da, self.input_mask = self.input_da[0], self.input_da[1]

    def norm_stats(self):
        if self._norm_stats is None:
            raise NormParamsNotProvided()
        return self._norm_stats

    def post_fn(self, phase):
        m, s = self.norm_stats()[phase]

        def normalize(item):
            return (item - m) / s

        return ft.partial(
            ft.reduce,
            lambda i, f: f(i),
            [
                TrainingItem._make,
                lambda item: item._replace(tgt=normalize(item.tgt)),
                lambda item: item._replace(input=normalize(item.input)),
            ],
        )


    def setup(self, stage="test"):
        self.train_ds = LazyXrDataset(
            self.input_da.sel(self.domains["train"]),
            **self.xrds_kw["train"],
            postpro_fn=self.post_fn("train"),
            mask=self.input_mask,
        )

        self.val_ds = LazyXrDataset(
            self.input_da.sel(self.domains["val"]),
            **self.xrds_kw["val"],
            postpro_fn=self.post_fn("val"),
            mask=self.input_mask,
        )

    #def val_dataloader(self):
    #    return torch.utils.data.DataLoader(
    #        self.val_ds,
    #        shuffle=False,
    #        batch_size=1,
    #        num_workers=1,
    #    )

class LazyXrDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ds,
        patch_dims,
        domain_limits=None,
        strides=None,
        postpro_fn=None,
        noise=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.return_coords = False
        self.postpro_fn = postpro_fn
        self.ds = ds.sel(**(domain_limits or {}))
        self.patch_dims = patch_dims
        self.strides = strides or {}
        _dims = ("variable",) + tuple(k for k in self.ds.dims)
        _shape = (2,) + tuple(self.ds[k].shape[0] for k in self.ds.dims)
        ds_dims = dict(zip(_dims, _shape))
        # ds_dims = dict(zip(self.ds.dims, self.ds.shape))
        self.ds_size = {
            dim: max(
                (ds_dims[dim] - patch_dims[dim]) // strides.get(dim, 1) + 1,
                0,
            )
            for dim in patch_dims
        }
        self._rng = np.random.default_rng()
        self.noise = noise
        self.mask = kwargs.get("mask")

    def __len__(self):
        size = 1
        for v in self.ds_size.values():
            size *= v
        return size

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_coords(self):
        self.return_coords = True
        coords = []
        try:
            for i in range(len(self)):
                coords.append(self[i])
        finally:
            self.return_coords = False
            return coords

    def __getitem__(self, item):
        sl = {}
        _zip = zip(
            self.ds_size.keys(), np.unravel_index(item, tuple(self.ds_size.values()))
        )

        for dim, idx in _zip:
            sl[dim] = slice(
                self.strides.get(dim, 1) * idx,
                self.strides.get(dim, 1) * idx + self.patch_dims[dim],
            )

        if self.mask is not None:
            start, stop = sl["time"].start % 365, sl["time"].stop % 365
            if start > stop:
                start -= stop
                stop = None
            sl_mask = sl.copy()
            sl_mask["time"] = slice(start, stop)

            da = self.ds.isel(**sl)

            item = (
                da.to_dataset(name="tgt")
                .assign(input=da.where(self.mask.isel(**sl_mask).values))
                .to_array()
                .sortby("variable")
            )
        else:
            item = self.ds.isel(**sl)

        if self.return_coords:
            return item.coords.to_dataset()[list(self.patch_dims)]

        item = item.data.astype(np.float32)

        if self.noise:
            noise = np.tile(
                self._rng.uniform(-self.noise, self.noise, item[0].shape), (2, 1, 1, 1)
            ).astype(np.float32)
            item = item + noise

        if self.postpro_fn is not None:
            return self.postpro_fn(item)
        return item

def load_glorys12_data(tgt_path, inp_path, tgt_var="zos", inp_var="input"):
    isel = None  # dict(time=slice(-465, -265))

    _start = time.time()

    
    print('..... Start loading dataset',flush=True)
    
    tgt = (
        xr.open_dataset(tgt_path)[tgt_var]
        .isel(isel)
    )
    inp = xr.open_dataset(inp_path)[inp_var].isel(isel)

    ds = (
        xr.Dataset(
            dict(input=inp, tgt=(tgt.dims, tgt.values)),
            inp.coords,
        )
        .to_array()
        .sortby("variable")
    )

    print(f">>> Durée de chargement : {time.time() - _start:.4f} s",flush=True)
    #print(ds,flush=True)

    #print(ds["tgt"].data,flush=True)

    return ds

def load_glorys12_data_on_fly_inp(
    tgt_path,
    inp_path,
    tgt_var="zos",
    inp_var="input",
):
 
    print('..... Start lazy loading',flush=True)

    isel = None  # dict(time=slice(-365 * 2, None))

    tgt = (
        xr.open_dataset(tgt_path)[tgt_var]
        .isel(isel)
        #.rename(latitude="lat", longitude="lon")
    )
    print('..... GT dataset')

    if list(tgt.coords)[1] == 'latitude' :
        tgt = tgt.rename(latitude="lat", longitude="lon")


    inp = (
        xr.open_dataset(inp_path)[inp_var]
        .isel(isel)
        #.rename(latitude="lat", longitude="lon")
    )

    if list(inp.coords)[1] == 'latitude' :
        inp = inp.rename(latitude="lat", longitude="lon")
    print('..... Obs dataset')

    return tgt, inp

def train(trainer, dm, lit_mod, ckpt=None):
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    start = time.time()
    
    print(" Parameter configurations for lit_mod")
    print(lit_mod.hparams)
    trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)


    print(f"Durée d'apprentissage : {time.time() - start:.3} s")

def load_from_cfg(cfg_path, key):
    """
    Load configurations from a specified file and instantiate the
    desired node.
    """
    cfg = OmegaConf.load(Path(cfg_path))
    node = OmegaConf.select(cfg, key)
    return hydra.utils.call(node)
