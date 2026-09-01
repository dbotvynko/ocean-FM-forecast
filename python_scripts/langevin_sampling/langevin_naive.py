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
from langevin_utils import langevin_sampling, write_langevin_results
sys.path.append('../../forecast/4dvarnet-starter-glorys12')


# IMPORT PRIOR
with initialize(version_base='1.3', config_path="../../forecast/4dvarnet-starter-glorys12/config"):
    base_cfg = compose(config_name='main', overrides=['xp=base_forecast_global_softedge_fastrec_GPU_1patch_10y'])

module_4dvar = hydra.utils.call(OmegaConf.select(base_cfg, 'model'))
ckpt_path = '/Odyssey/public/glorys/trainings/forecast_1patch_10y/base_forecast_global_softedge_fastrec_GPU_1patch_10y/checkpoints/val_mse=6.89547-epoch=072.ckpt'
module_4dvar.load_state_dict(torch.load(ckpt_path, map_location='cuda')['state_dict'])

######### module_4dvar.train()
module_4dvar.eval()

prior = module_4dvar.solver.prior_cost.cuda()


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


# CREATE STARTING STATE: x_0
data_path_out_f = '/Odyssey/public/glorys/rec/glorys4_global_1patch_fulloutput/nrt_2023_global_4/test_data_{}.nc'
x_array = []
for i, t in enumerate(time_array):
    x_array.append(torch.Tensor((xr.open_dataset(data_path_out_f.format(i)).sel(time=t)['out'].values - norms[0]) / norms[1]).type(torch.FloatTensor))
x_0 = torch.stack(x_array, dim=0)


# CREATE MASKING ARRAY
array_mask = torch.Tensor(xr.open_dataset('/Odyssey/public/glorys/reanalysis/glorys12_2020_4th.nc').sel(time='2020-01-01').notnull()['zos'].drop_vars('depth').values).type(torch.bool).unsqueeze(0).tile((29,1,1))


# PERFORM GRID_SEARCH
write_path = '/Odyssey/public/glorys/langevin_sampling/tests_3/'
num_samples = 5000
alpha = 1
step_sizes = [0.1, 0.01, 0.001, 0.0001]
noise_scalings = [0.1, 0.01, 0.001, 0.0001]

for step_size in tqdm(step_sizes, desc='step size:', position=0):
    for noise_scaling in tqdm(noise_scalings, desc='noise scaling:', position=1):
        tqdm.write('step_size: {} | noise_scaling: {}|'.format(step_size, noise_scaling))

        langevin_results = langevin_sampling(x_0, y, prior, array_mask, num_samples=num_samples, step_size=step_size, alpha=alpha, noise_scaling=noise_scaling)
        langevin_results['step_size'] = step_size
        langevin_results['noise_scaling'] = noise_scaling
        write_langevin_results(write_path, langevin_results)