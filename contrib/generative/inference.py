"""
Standalone inference utilities for GenFlowLit models.

This exists separately from models.py because, for the current xp configs,
GenFlowLit.test_step/on_test_epoch_end are unused stubs and
DistinctNormDataModule never builds a test_ds/test_dataloader: there is no
trainer.test()/predict() path. Real inference is done by loading a trained
checkpoint and calling GenFlowLit.sample(...) directly on a hand-built
batch, which is what the functions/classes below wrap for the "forecast a
gridded SLA product, score per lead time" use case (e.g. 2023 NRT nadirs).

Nothing here modifies existing classes/functions in models.py or glorys12.
"""

from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

PATCH_TIME_DEFAULT = 29

InferenceItem = namedtuple("InferenceItem", ["input"])


def load_gridded_sla(path, var="sla_unfiltered", lat_slice=None, lon_slice=None):
    """
    Load a gridded SLA product, renaming lat/lon if needed and optionally
    cropping to the (lat_slice, lon_slice) domain the model was trained on
    (e.g. the raw product may cover the full global grid while the model
    expects the domain.train crop).
    """
    ds = xr.open_dataset(path)
    if list(ds.coords)[1] == "latitude":
        ds = ds.rename(latitude="lat", longitude="lon")
    da = ds[var]
    if lat_slice is not None or lon_slice is not None:
        da = da.sel(lat=lat_slice or slice(None), lon=lon_slice or slice(None))
    return da


def load_gen_flow_checkpoint(model, ckpt_path, device="cuda"):
    """Load trained weights into an already-instantiated GenFlowLit and set eval mode."""
    state_dict = torch.load(ckpt_path, map_location=device)["state_dict"]
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def build_forecast_window(sla_da, start_date, norm_stats, patch_time=PATCH_TIME_DEFAULT):
    """
    Build a normalized (1, patch_time, lat, lon) input tensor for the window
    starting at `start_date`, plus the raw (un-normalized) window DataArray
    used as evaluation truth.
    """
    m, s = norm_stats
    dates = pd.date_range(start_date, periods=patch_time, freq="D")
    window = sla_da.sel(time=dates)
    values = (window.values - m) / s
    tensor = torch.from_numpy(np.asarray(values)).float().unsqueeze(0)
    return InferenceItem(input=tensor), window


def leadtime_indices(patch_time=PATCH_TIME_DEFAULT, leadtimes=range(7)):
    """Map lead days (0 = first forecast day) to indices in the patch_time window."""
    obs_days = patch_time // 2
    return {lt: obs_days + lt for lt in leadtimes}


@torch.no_grad()
def forecast_window(model, item, denorm_stats):
    """Run GenFlowLit.sample on one window, return the final denormalized (patch_time, lat, lon) field."""
    m, s = denorm_stats
    samples = model.sample(item)
    final = samples[-1].squeeze(0).cpu().numpy()
    return final * s + m


class YearlyLeadtimeEvaluator:
    """
    Slides a patch_time-day window one day at a time over a gridded SLA
    product, forecasts each window with GenFlowLit.sample, and scores the
    requested lead times (0 = first forecast day) against the same
    product's own values at that date (sparse, wherever it has data).

    One sample per window by default (num_samples=1); increase for an
    ensemble later without changing anything else here.
    """

    def __init__(
        self,
        model,
        sla_da,
        norm_stats,
        patch_time=PATCH_TIME_DEFAULT,
        leadtimes=range(7),
        num_samples=1,
    ):
        self.model = model
        self.sla_da = sla_da
        self.norm_stats = norm_stats
        self.patch_time = patch_time
        self.leadtimes = list(leadtimes)
        self.num_samples = num_samples
        self.lt_to_idx = leadtime_indices(patch_time, self.leadtimes)

    def run_day(self, start_date):
        item, window = build_forecast_window(
            self.sla_da, start_date, self.norm_stats, self.patch_time
        )
        truth = window.values

        forecasts = [
            forecast_window(self.model, item, self.norm_stats)
            for _ in range(self.num_samples)
        ]
        forecasts = np.stack(forecasts, axis=0)  # (num_samples, patch_time, lat, lon)

        result = {}
        for lt, idx in self.lt_to_idx.items():
            pred = forecasts[:, idx]  # (num_samples, lat, lon)
            true = truth[idx]
            finite = np.isfinite(true)
            pred_mean = pred.mean(axis=0)
            rmse = (
                np.sqrt(np.nanmean((pred_mean[finite] - true[finite]) ** 2))
                if finite.any()
                else np.nan
            )
            result[lt] = dict(pred=pred, true=true, rmse=rmse)
        return result

    def day_result_to_dataset(self, start_date, day_result):
        """
        Pack one run_day() result into an xr.Dataset with real lat/lon and
        leadtime/valid_time coordinates:
          - forecast(sample, leadtime, lat, lon)
          - truth(leadtime, lat, lon)
          - rmse(leadtime)
        """
        start_date = pd.Timestamp(start_date)
        obs_days = self.patch_time // 2
        lat = self.sla_da.lat.values
        lon = self.sla_da.lon.values

        forecast = np.stack([day_result[lt]["pred"] for lt in self.leadtimes], axis=1)
        truth = np.stack([day_result[lt]["true"] for lt in self.leadtimes], axis=0)
        rmse = np.array([day_result[lt]["rmse"] for lt in self.leadtimes])
        valid_time = [start_date + pd.Timedelta(days=int(obs_days + lt)) for lt in self.leadtimes]

        return xr.Dataset(
            data_vars=dict(
                forecast=(("sample", "leadtime", "lat", "lon"), forecast),
                truth=(("leadtime", "lat", "lon"), truth),
                rmse=(("leadtime",), rmse),
            ),
            coords=dict(
                sample=np.arange(self.num_samples),
                leadtime=list(self.leadtimes),
                valid_time=("leadtime", valid_time),
                lat=lat,
                lon=lon,
                init_time=start_date,
            ),
        )

    def run_year(self, start_dates, out_dir=None):
        rmses = {lt: [] for lt in self.leadtimes}

        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

        for start_date in start_dates:
            day_result = self.run_day(start_date)
            for lt in self.leadtimes:
                rmses[lt].append(day_result[lt]["rmse"])

            if out_dir is not None:
                out_path = out_dir / f"{pd.Timestamp(start_date).date()}.nc"
                self.day_result_to_dataset(start_date, day_result).to_netcdf(out_path)

        return rmses
