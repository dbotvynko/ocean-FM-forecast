"""
Copy the part of a (large, multi-variable) NetCDF file that a training reads
to the node-local /SCRATCH, so the training reads a compact local file
instead of the shared /Odyssey NFS. Does nothing if the target already exists.

Only one variable, a time range and a latitude range are kept (e.g. the L3
target/input of the OSE L3-target xp: sla_unfiltered, 2010-2019, -80..90N is
~14 GB instead of the 106 GB 2010-2023 multi-variable file). The copy is
written to <target>.tmp and renamed at the end, so an interrupted copy is
never mistaken for a complete one.

/SCRATCH is local to each node: run this at the start of the job, on the
node the training runs on.

Usage:
    python stage_to_scratch.py SRC DST --var VAR --time-start 2010-01-01
        --time-end 2019-12-31 [--lat-min -80 --lat-max 90]
"""

import argparse
import time
from pathlib import Path

import dask
import xarray as xr

# Single-threaded: dask's threaded scheduler reading from and writing to
# NetCDF/HDF5 at the same time can deadlock (all threads stuck on the HDF5
# lock, seen on a full 10-year copy); copying chunk by chunk avoids it.
dask.config.set(scheduler="synchronous")

parser = argparse.ArgumentParser()
parser.add_argument("src")
parser.add_argument("dst")
parser.add_argument("--var", required=True)
parser.add_argument("--time-start", required=True)
parser.add_argument("--time-end", required=True)
parser.add_argument("--lat-min", type=float, default=None)
parser.add_argument("--lat-max", type=float, default=None)
args = parser.parse_args()

dst = Path(args.dst)
if dst.exists():
    print(f"{dst} already on this node's /SCRATCH, nothing to copy", flush=True)
    raise SystemExit(0)

dst.parent.mkdir(parents=True, exist_ok=True)
tmp = dst.with_name(dst.name + ".tmp")
start = time.time()

ds = xr.open_dataset(args.src, chunks={"time": 100})
if "latitude" in ds.dims:
    ds = ds.rename(latitude="lat", longitude="lon")
sub = ds[[args.var]].sel(time=slice(args.time_start, args.time_end),
                         lat=slice(args.lat_min, args.lat_max))
print(f"copying {args.var} {dict(sub.sizes)} from {args.src} to {dst} ...", flush=True)
sub.to_netcdf(tmp)
tmp.replace(dst)
print(f"done in {(time.time() - start) / 60:.1f} min ({dst.stat().st_size / 1e9:.1f} GB)", flush=True)
