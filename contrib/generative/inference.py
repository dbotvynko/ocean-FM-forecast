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


def crps_ensemble(ensemble, truth):
    """
    Empirical CRPS (energy/NRG estimator) for a finite ensemble forecast
    against an observation:

        CRPS = (1/N) sum_i |x_i - y|  -  (1/(2N^2)) sum_i sum_j |x_i - x_j|

    ensemble: (N, ...) array of ensemble members.
    truth: array broadcastable against ensemble's trailing dims.
    Returns an array shaped like truth (NaN wherever truth is NaN).

    This needs the raw ensemble members, so it can only be computed while
    they're still in memory (e.g. inside YearlyLeadtimeEvaluator.run_day),
    not from an already-saved forecast_mean/forecast_std file -- see
    gaussian_crps for that case.

    Known bias: this estimator systematically overestimates CRPS for
    small N (verified: N=5 gave ~0.35 vs the true ~0.23 for a standard
    normal in a quick check here), converging to the true value as N
    grows. Fine for comparing conditions that share the same N (e.g.
    leadtime-to-leadtime, or model-to-model at the same ensemble size),
    but don't compare its absolute value across different N without
    correcting for this (the "fair CRPS" variant divides the second term
    by N*(N-1) instead of N^2, which removes the bias at the cost of
    higher variance for small N).
    """
    n = ensemble.shape[0]
    term1 = np.abs(ensemble - truth[None, ...]).mean(axis=0)
    diffs = np.abs(ensemble[:, None, ...] - ensemble[None, :, ...])
    term2 = diffs.sum(axis=(0, 1)) / (2 * n * n)
    return term1 - term2


def crps_ensemble_fair(ensemble, truth):
    """
    Bias-corrected ("fair") CRPS (Ferro et al. 2008): identical to
    crps_ensemble except the second term is normalized by N*(N-1)
    instead of N^2, which removes the small-N upward bias analytically
    -- no larger ensemble needed. Verified here: N=5 gives ~0.232 vs the
    true ~0.231 for a standard normal (500-trial average), essentially
    unbiased at every N tested (5, 20, 100, 1000), unlike crps_ensemble's
    ~0.33 at N=5. Use this one; crps_ensemble is kept for reference/
    comparison against the naive estimator.
    """
    n = ensemble.shape[0]
    term1 = np.abs(ensemble - truth[None, ...]).mean(axis=0)
    diffs = np.abs(ensemble[:, None, ...] - ensemble[None, :, ...])
    term2 = diffs.sum(axis=(0, 1)) / (2 * n * (n - 1))
    return term1 - term2


def gaussian_crps(mean, std, truth):
    """
    Closed-form CRPS under a Gaussian approximation N(mean, std) of the
    forecast distribution (Gneiting & Raftery 2007), usable when only
    summary statistics are available (e.g. this repo's saved
    forecast_mean/forecast_std, not the raw ensemble members):

        z = (truth - mean) / std
        CRPS = std * [ z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]

    An approximation, not the exact empirical CRPS -- see crps_ensemble
    for that (requires the raw ensemble, computed on the fly).
    """
    from scipy.stats import norm

    z = (truth - mean) / std
    return std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


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
            attrs=dict(obs_days=obs_days),
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

    def day_result_to_mean_std_dataset(self, start_date, day_result):
        """
        Like day_result_to_dataset, but collapses the sample dimension:
        saves the ensemble mean forecast plus its per-pixel std map
        instead of every raw member, since num_samples>1 raw members
        would multiply the on-disk size by num_samples for no benefit
        once you only care about the ensemble summary.
          - forecast_mean(leadtime, lat, lon)
          - forecast_std(leadtime, lat, lon)
          - truth(leadtime, lat, lon)
          - rmse(leadtime)
        """
        start_date = pd.Timestamp(start_date)
        obs_days = self.patch_time // 2
        lat = self.sla_da.lat.values
        lon = self.sla_da.lon.values

        forecast_mean = np.stack(
            [day_result[lt]["pred"].mean(axis=0) for lt in self.leadtimes], axis=0
        )
        forecast_std = np.stack(
            [day_result[lt]["pred"].std(axis=0) for lt in self.leadtimes], axis=0
        )
        truth = np.stack([day_result[lt]["true"] for lt in self.leadtimes], axis=0)
        rmse = np.array([day_result[lt]["rmse"] for lt in self.leadtimes])
        valid_time = [start_date + pd.Timedelta(days=int(obs_days + lt)) for lt in self.leadtimes]

        return xr.Dataset(
            data_vars=dict(
                forecast_mean=(("leadtime", "lat", "lon"), forecast_mean.astype(np.float32)),
                forecast_std=(("leadtime", "lat", "lon"), forecast_std.astype(np.float32)),
                truth=(("leadtime", "lat", "lon"), truth.astype(np.float32)),
                rmse=(("leadtime",), rmse.astype(np.float32)),
            ),
            coords=dict(
                leadtime=list(self.leadtimes),
                valid_time=("leadtime", valid_time),
                lat=lat,
                lon=lon,
                init_time=start_date,
            ),
            attrs=dict(obs_days=obs_days, num_samples=self.num_samples),
        )

    def run_year_mean_std(self, start_dates, out_dir):
        """
        Like run_year, but writes day_result_to_mean_std_dataset's output
        (mean + std, no per-sample data) instead of every raw member, with
        zlib compression on by default since that's the whole point of
        using this instead of run_year with a larger num_samples.
        """
        rmses = {lt: [] for lt in self.leadtimes}

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for start_date in start_dates:
            day_result = self.run_day(start_date)
            for lt in self.leadtimes:
                rmses[lt].append(day_result[lt]["rmse"])

            out_path = out_dir / f"{pd.Timestamp(start_date).date()}.nc"
            ds = self.day_result_to_mean_std_dataset(start_date, day_result)
            encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
            ds.to_netcdf(out_path, encoding=encoding)

        return rmses

    def day_result_to_mean_std_crps_dataset(self, start_date, day_result):
        """
        Like day_result_to_mean_std_dataset, but also computes CRPS per
        leadtime/pixel from the raw ensemble in day_result before it's
        discarded (both the naive and bias-corrected "fair" estimators --
        see crps_ensemble / crps_ensemble_fair). CRPS needs the full
        ensemble, not just mean/std, so it can only be computed here --
        not retroactively from an already-saved mean/std-only file (use
        gaussian_crps against forecast_mean/forecast_std for that case).
          - forecast_mean(leadtime, lat, lon)
          - forecast_std(leadtime, lat, lon)
          - crps(leadtime, lat, lon)        naive estimator, biased for small N
          - crps_fair(leadtime, lat, lon)   bias-corrected -- use this one
          - truth(leadtime, lat, lon)
          - rmse(leadtime)
        """
        start_date = pd.Timestamp(start_date)
        obs_days = self.patch_time // 2
        lat = self.sla_da.lat.values
        lon = self.sla_da.lon.values

        forecast_mean = np.stack(
            [day_result[lt]["pred"].mean(axis=0) for lt in self.leadtimes], axis=0
        )
        forecast_std = np.stack(
            [day_result[lt]["pred"].std(axis=0) for lt in self.leadtimes], axis=0
        )
        crps = np.stack(
            [crps_ensemble(day_result[lt]["pred"], day_result[lt]["true"]) for lt in self.leadtimes],
            axis=0,
        )
        crps_fair = np.stack(
            [crps_ensemble_fair(day_result[lt]["pred"], day_result[lt]["true"]) for lt in self.leadtimes],
            axis=0,
        )
        truth = np.stack([day_result[lt]["true"] for lt in self.leadtimes], axis=0)
        rmse = np.array([day_result[lt]["rmse"] for lt in self.leadtimes])
        valid_time = [start_date + pd.Timedelta(days=int(obs_days + lt)) for lt in self.leadtimes]

        return xr.Dataset(
            data_vars=dict(
                forecast_mean=(("leadtime", "lat", "lon"), forecast_mean.astype(np.float32)),
                forecast_std=(("leadtime", "lat", "lon"), forecast_std.astype(np.float32)),
                crps=(("leadtime", "lat", "lon"), crps.astype(np.float32)),
                crps_fair=(("leadtime", "lat", "lon"), crps_fair.astype(np.float32)),
                truth=(("leadtime", "lat", "lon"), truth.astype(np.float32)),
                rmse=(("leadtime",), rmse.astype(np.float32)),
            ),
            coords=dict(
                leadtime=list(self.leadtimes),
                valid_time=("leadtime", valid_time),
                lat=lat,
                lon=lon,
                init_time=start_date,
            ),
            attrs=dict(obs_days=obs_days, num_samples=self.num_samples),
        )

    def run_year_mean_std_crps(self, start_dates, out_dir):
        """
        Like run_year_mean_std, but also writes the exact ensemble CRPS
        per leadtime/pixel (day_result_to_mean_std_crps_dataset) -- needs
        num_samples > 1 to be meaningful (CRPS of a 1-member "ensemble"
        degenerates to plain absolute error).
        """
        rmses = {lt: [] for lt in self.leadtimes}

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for start_date in start_dates:
            day_result = self.run_day(start_date)
            for lt in self.leadtimes:
                rmses[lt].append(day_result[lt]["rmse"])

            out_path = out_dir / f"{pd.Timestamp(start_date).date()}.nc"
            ds = self.day_result_to_mean_std_crps_dataset(start_date, day_result)
            encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
            ds.to_netcdf(out_path, encoding=encoding)

        return rmses


def combine_leadtime_files(out_dir, leadtimes=range(7)):
    """
    Combine the per-day .nc files written by YearlyLeadtimeEvaluator.run_year
    (each holding all leadtimes for one window) into one file per lead time,
    concatenated over every day found in out_dir along a new `time`
    dimension (each day's valid_time for that lead time).

    Output files are named test_leadtime_<idx>.nc, where <idx> is the
    absolute window index (obs_days + leadtime, e.g. 14 for leadtime 0,
    20 for leadtime 6) -- the same convention already used elsewhere in
    this repo (perturb_gen_eval.py's `for LEADTIME in range(14, 21)`).

    Safe to call at any point during a run_year (e.g. in another job) --
    it just picks up whichever per-day files exist in out_dir so far, so you
    don't have to wait for the full year to inspect leadtime 0 across the
    days already computed. Re-run once the year finishes for the complete
    set of files.
    """
    out_dir = Path(out_dir)
    day_paths = sorted(out_dir.glob("????-??-??.nc"))

    written = {}
    for lt in leadtimes:
        slices = []
        obs_days = None
        for p in day_paths:
            with xr.open_dataset(p) as ds:
                obs_days = ds.attrs["obs_days"]
                # NB: no drop=True here -- it would also drop valid_time,
                # since it's only indexed along the leadtime dimension and
                # becomes scalar as a side effect of this selection too.
                day_slice = ds.sel(leadtime=lt).load()
            # NB: NOT .item() -- numpy's datetime64[ns].item() returns a
            # plain int (ns since epoch), since Python's datetime.datetime
            # can't hold nanosecond precision. That silently produced an
            # integer `time` coordinate instead of datetime64, breaking
            # `.dt` accessors downstream. .values + pd.Timestamp keeps it
            # a real datetime.
            valid_time = pd.Timestamp(day_slice["valid_time"].values)
            slices.append(
                day_slice.drop_vars(["valid_time", "leadtime"]).expand_dims(time=[valid_time])
            )

        combined = xr.concat(slices, dim="time").sortby("time")
        combined = combined.assign_coords(time=pd.DatetimeIndex(combined.time.values))
        window_idx = obs_days + lt
        out_path = out_dir / f"test_leadtime_{window_idx}.nc"
        combined.to_netcdf(out_path)
        written[lt] = out_path

    return written


def gauss_seidel_fill_2d(values, max_iterations=2000, epsilon=1e-4, relaxation=1.0):
    """
    Fill NaN gaps in a 2D array via Gauss-Seidel relaxation, using the
    current pyinterp.fill.gauss_seidel API (operates on a plain ndarray
    in-place, returns (iterations, residual) -- not the Grid2D-wrapper
    API that python_scripts/utils/data_utils.py's remove_nan targets,
    which is stale against pyinterp>=2026.x installed in this env).
    """
    import pyinterp.fill

    grid = np.array(values, dtype=np.float64, copy=True)
    pyinterp.fill.gauss_seidel(
        grid, max_iterations=max_iterations, epsilon=epsilon, relaxation=relaxation
    )
    return grid


def estimate_obs_conditioned_residual_variance(tgt_da, mask_da, norm_stats, num_days=30, seed=0):
    """
    Empirical proxy for the irreducible val_loss floor near t=0 -- i.e.
    Var(tgt | sparse obs), the part of the loss no model can beat no matter
    how well trained, since at t=0 the network sees x0 exactly but nothing
    about tgt beyond what the observation network constrains.

    Reconstructs `num_days` randomly sampled days from their sparse 6-nadir
    mask using Gauss-Seidel gap filling (gauss_seidel_fill_2d above), and
    reports the residual MSE against the true field, in the same
    normalized units as the training loss (divided by
    datamodule.norm_stats.train's std, like tgt/input are normalized
    before reaching GenFlowLit).

    This is only a proxy, not the model's actual achievable floor: a
    trained network may do better (it can use learned spatio-temporal
    priors, not just a smoothness prior) or worse (if underfit). It gives
    a concrete, reproducible order of magnitude rather than a guess.
    """
    m, s = norm_stats
    rng = np.random.default_rng(seed)
    n_days = tgt_da.sizes["time"]
    n_mask_days = mask_da.sizes["time"]
    sample_idx = rng.choice(n_days, size=min(num_days, n_days), replace=False)

    sq_errors = []
    for i in sample_idx:
        truth = tgt_da.isel(time=int(i)).load()
        day_mask = mask_da.isel(time=int(i) % n_mask_days)
        obs_values = truth.where(day_mask.values).values

        filled_values = gauss_seidel_fill_2d(obs_values)

        truth_n = (truth.values - m) / s
        filled_n = (filled_values - m) / s
        sq_errors.append((filled_n - truth_n) ** 2)

    sq_errors = np.concatenate([e.ravel() for e in sq_errors])
    return dict(
        mean_sq_error=float(np.nanmean(sq_errors)),
        rmse_normalized=float(np.sqrt(np.nanmean(sq_errors))),
        num_days=len(sample_idx),
    )


def estimate_persistence_forecast_floor(
    tgt_da,
    mask_da,
    norm_stats,
    leadtimes=range(7),
    num_windows=30,
    seed=0,
    patch_time=PATCH_TIME_DEFAULT,
):
    """
    Persistence-forecast baseline: reconstruct the last observed day (via
    its own sparse mask + Gauss-Seidel gap fill) and forecast every
    requested lead time as that same, unchanged field. This is the
    standard reference in forecast verification ("skill relative to
    persistence") -- unlike estimate_obs_conditioned_residual_variance, it
    does use temporal information (the trajectory up to the last observed
    day informs which day gets fixed and filled), it's just naive about
    how the field evolves afterward (assumes no change).

    Uses the same window/leadtime convention as YearlyLeadtimeEvaluator
    (obs_days = patch_time // 2, leadtime 0 = first forecast day), so its
    per-leadtime RMSEs are directly comparable to a model's
    test_leadtime_<idx>.nc rmse values.
    """
    m, s = norm_stats
    rng = np.random.default_rng(seed)
    obs_days = patch_time // 2
    n_days = tgt_da.sizes["time"]
    n_mask_days = mask_da.sizes["time"]

    max_start = n_days - patch_time
    start_idx = rng.choice(max_start + 1, size=min(num_windows, max_start + 1), replace=False)

    sq_errors = {lt: [] for lt in leadtimes}
    for start in start_idx:
        last_obs_idx = int(start) + obs_days - 1
        last_obs = tgt_da.isel(time=last_obs_idx).load()
        last_obs_mask = mask_da.isel(time=last_obs_idx % n_mask_days)
        obs_values = last_obs.where(last_obs_mask.values).values

        persisted_n = (gauss_seidel_fill_2d(obs_values) - m) / s

        for lt in leadtimes:
            target_idx = int(start) + obs_days + lt
            truth_n = (tgt_da.isel(time=target_idx).values - m) / s
            sq_errors[lt].append((persisted_n - truth_n) ** 2)

    result = {}
    for lt in leadtimes:
        errs = np.concatenate([e.ravel() for e in sq_errors[lt]])
        result[lt] = dict(
            mean_sq_error=float(np.nanmean(errs)),
            rmse_normalized=float(np.sqrt(np.nanmean(errs))),
        )
    return result
