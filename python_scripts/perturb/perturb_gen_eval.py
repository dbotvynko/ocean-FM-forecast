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
from sampling_eval_utils import get_sampling_eval_dict, write_eval_dict, delete_sample_files
sys.path.append('../../forecast/4dvarnet-starter-glorys12')


# IMPORT 4DVAR
with initialize(version_base='1.3', config_path="../../forecast/4dvarnet-starter-glorys12/config"):
    base_cfg = compose(config_name='main', overrides=['xp=base_forecast_global_softedge_fastrec_GPU_1patch_10y'])

module_4dvar = hydra.utils.call(OmegaConf.select(base_cfg, 'model')).cuda()
ckpt_path = '/Odyssey/public/glorys/trainings/forecast_1patch_10y/base_forecast_global_softedge_fastrec_GPU_1patch_10y/checkpoints/val_mse=6.89547-epoch=072.ckpt'
module_4dvar.load_state_dict(torch.load(ckpt_path, map_location='cuda')['state_dict'])
module_4dvar.eval()


# TIME WINDOW
time = '2023-01-01'
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

array_mask = torch.Tensor(xr.open_dataset('/Odyssey/public/glorys/reanalysis/glorys12_2020_4th.nc').sel(time='2020-01-01').notnull()['zos'].drop_vars('depth').values).type(torch.bool).unsqueeze(0).tile((29,1,1))

# PERFORM PERTURB
write_path = '/Odyssey/public/glorys/perturb_sampling/gen_y1_{}/'

num_samples = 100
noise_scaling = 0.08

# iterate over whole year
for d in tqdm(range(365-29), desc='days', position=0):

    time_d_ = time_ + timedelta(days=1)*d
    time_array = [(time_d_ + i * timedelta(days=1)).strftime(format='%Y-%m-%d') for i in range(29)]
    y = (torch.Tensor(y_raw_ds.sel(time=time_array).values) - norms[0]) / norms[1]
    denorm_y = y * norms[1] + norms[0]

    # generate num_samples samples
    for i in tqdm(range(num_samples), desc='sampling', position=1):

        perturb_results = perturb_sampling(y, module_4dvar, noise_scaling=noise_scaling)
        
        write_perturb_results(write_path.format(time_d_.strftime(format='%Y-%m-%d')), i, perturb_results)

    files = write_path+'sample_{}/sample.pickle'

    for LEADTIME in range(14,21):

        samples = list()
        for i in range(100):
            try:
                with open(files.format(time_d_.strftime(format='%Y-%m-%d'), i), 'rb') as f:
                    samples.append(pickle.load(f)[LEADTIME])
                    f.close()
            except:
                break

        samples = np.array(samples)
        samples = samples * norms[1] + norms[0]

        write_dict = get_sampling_eval_dict(samples, denorm_y[LEADTIME])
        write_eval_dict(write_dict, write_path.format(time_d_.strftime(format='%Y-%m-%d')), 'lt_{}'.format(LEADTIME))
    delete_sample_files(write_path.format(time_d_.strftime(format='%Y-%m-%d')), num_samples)
