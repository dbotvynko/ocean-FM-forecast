import os
import shutil
import pickle

import numpy as np
import torch

from pysteps.verification.probscores import CRPS

def get_sampling_eval_dict(samples, y):

    averaged_sample = np.mean(samples, axis=0)
    std_sample = np.std(samples, axis=0)

    obs_mask = y.isfinite()

    average_y_rmse = torch.sqrt(torch.mean((torch.Tensor(averaged_sample)[obs_mask] - (y)[obs_mask])**2))

    rmses_samples = []
    for sample in samples:
        rmses_samples.append(torch.sqrt(torch.mean((torch.Tensor(sample)[obs_mask] - (y)[obs_mask])**2)))


    average_y_mae = torch.mean(torch.abs(torch.Tensor(averaged_sample)[obs_mask] - (y)[obs_mask]))

    maes_samples = []
    for sample in samples:
        maes_samples.append(torch.mean(torch.abs(torch.Tensor(sample)[obs_mask] - (y)[obs_mask])))

    crps = CRPS(samples, np.array(y))

    eval_dict = dict(
        average_sample = averaged_sample,
        var_sample = std_sample,
        rmse_average = average_y_rmse,
        rmse_samples = rmses_samples,
        mae_average = average_y_mae,
        mae_samples = maes_samples,
        crps = crps,
    )

    return eval_dict


def write_eval_dict(eval_dict, write_path, subfolder_name):
    
    write_path = os.path.join(write_path, subfolder_name)
    os.makedirs(write_path)
    folder_path_f = os.path.join(write_path, '{}.pickle')

    for key, value in eval_dict.items():
        with open(folder_path_f.format(key), 'wb') as f:
            pickle.dump(value, f)

def delete_sample_files(write_path, num_samples):
    write_path = write_path+'sample_{}/'
    files_paths = [write_path.format(i) for i in range(num_samples)]

    for file_path in files_paths:
        shutil.rmtree(file_path)