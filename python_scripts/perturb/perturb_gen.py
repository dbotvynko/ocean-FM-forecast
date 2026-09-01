import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
import hydra
from hydra import initialize, compose
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import sys
from tqdm import tqdm
from datetime import datetime
from datetime import timedelta
import pickle

from perturb_utils import perturb_sampling, write_perturb_results
sys.path.append('../../forecast/4dvarnet-starter-glorys12')


# IMPORT 4DVAR
with initialize(version_base='1.3', config_path="../../forecast/4dvarnet-starter-glorys12/config"):
    base_cfg = compose(config_name='main', overrides=['xp=base_forecast_global_softedge_fastrec_GPU_1patch_10y'])

module_4dvar = hydra.utils.call(OmegaConf.select(base_cfg, 'model'))
ckpt_path = '/Odyssey/public/glorys/trainings/forecast_1patch_10y/base_forecast_global_softedge_fastrec_GPU_1patch_10y/checkpoints/val_mse=6.89547-epoch=072.ckpt'
module_4dvar.load_state_dict(torch.load(ckpt_path, map_location='cuda')['state_dict'])
module_4dvar.eval()


# TIME WINDOW
time = '2023-03-01'
time_ = datetime.strptime(time, '%Y-%m-%d')
time_array = [(time_ + i * timedelta(days=1)).strftime(format='%Y-%m-%d') for i in range(29)]


# CREATE OBS: y
data_path_y = '/Odyssey/public/altimetry_traces/nrt_2023_global_4/gridded/gridded_input.nc'
y_raw_ds = xr.open_dataset(data_path_y)['ssh']
y_raw = y_raw_ds.values

norms = []
norms.append(np.nanmean(y_raw))
norms.append(np.nanstd(y_raw))
print('norms: {:.4f} | {:.4f}'.format(norms[0], norms[1]))

y = (torch.Tensor(y_raw_ds.sel(time=time_array).values) - norms[0]) / norms[1]

# PERFORM PERTURB
write_path = '/Odyssey/public/glorys/perturb_sampling/gen1_{}/'
num_samples = 200

noise_scaling = 0.01

n_runs_langevin = 100

for i in range(n_runs_langevin):

    perturb_results = perturb_sampling(y, module_4dvar, num_samples=num_samples, noise_scaling=noise_scaling)
    perturb_results['noise_scaling'] = noise_scaling
    
    write_perturb_results(write_path.format(i), perturb_results)
