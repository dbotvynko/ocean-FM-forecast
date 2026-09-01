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
from langevin_utils import langevin_sampling, write_langevin_results, mvh_corr, cross_corr, crps_image_iterative
sys.path.append('../../forecast/4dvarnet-starter-glorys12')


# IMPORT PRIOR
with initialize(version_base='1.3', config_path="../../forecast/4dvarnet-starter-glorys12/config"):
    base_cfg = compose(config_name='main', overrides=['xp=base_forecast_global_softedge_fastrec_GPU_1patch_10y'])

module_4dvar = hydra.utils.call(OmegaConf.select(base_cfg, 'model'))
ckpt_path = '/Odyssey/public/glorys/trainings/forecast_1patch_10y/base_forecast_global_softedge_fastrec_GPU_1patch_10y/checkpoints/val_mse=6.89547-epoch=072.ckpt'
module_4dvar.load_state_dict(torch.load(ckpt_path, map_location='cuda')['state_dict'])
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
#x_array = []
#for i, t in enumerate(time_array):
#    x_array.append(torch.Tensor((xr.open_dataset(data_path_out_f.format(i)).sel(time=t)['out'].values - norms[0]) / norms[1]).type(torch.FloatTensor))
#x_0 = torch.stack(x_array, dim=0)

x_path = '/Odyssey/public/glorys/perturb_sampling/gen0_0/sampling_0/sample.pickle'
with open(x_path, 'rb') as f:
    x_0 = torch.Tensor(pickle.load(f)).type(torch.FloatTensor)
    f.close()

# CREATE MASKING ARRAY
array_mask = torch.Tensor(xr.open_dataset('/Odyssey/public/glorys/reanalysis/glorys12_2020_4th.nc').sel(time='2020-01-01').notnull()['zos'].drop_vars('depth').values).type(torch.bool).unsqueeze(0).tile((29,1,1))


# PERFORM LANGEVIN
write_path = '/Odyssey/public/glorys/langevin_sampling/gen4_{}/'
num_samples = 500
alpha = 0.0000001
step_size = 0.01
noise_scaling = 0.001

n_runs_langevin = 100

for i in range(n_runs_langevin):

    langevin_results = langevin_sampling(x_0, y, prior, array_mask, num_samples=num_samples, step_size=step_size, alpha=alpha, noise_scaling=noise_scaling, return_samples=False, return_last_sample=True)
    langevin_results['step_size'] = step_size
    langevin_results['noise_scaling'] = noise_scaling
    
    write_langevin_results(write_path.format(i), langevin_results)
