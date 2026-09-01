import numpy as np
import torch
from collections import namedtuple
import pickle
import os

TrainingItem = namedtuple('TrainingItem', ['input', 'tgt'])

def perturb_sampling(y, module_4dvar, noise_scaling, device='cuda'):

    y_zeros = torch.zeros_like(y)
    noise_mean = y_zeros
    noise_std = torch.full_like(y_zeros, noise_scaling)
    noise = torch.normal(noise_mean, noise_std)

    y_noised = torch.where(y.isfinite(), y+noise, np.NAN).unsqueeze(dim=0).to(device=device)
    y_noised[:,14:] = np.NAN

    batch = TrainingItem(
        input=y_noised,
        tgt=y_noised
    )

    out = np.array(module_4dvar(batch=batch).cpu().detach().squeeze())

    perturb_result = dict(
        sample=out,
        noise_scaling=noise_scaling
    )

    return perturb_result

#def write_perturb_results(path, perturb_results):
#    folder_name = 'sampling'
#    folder_path = os.path.join(path, '{}_0'.format(folder_name))
#    counter = 0
#    while os.path.exists(folder_path):
#        counter+=1
#        folder_path = os.path.join(path, '{}_{}'.format(folder_name, counter))
#    os.makedirs(folder_path)
#    folder_path_f = os.path.join(folder_path, '{}.pickle')
#
#    for key, value in perturb_results.items():
#        with open(folder_path_f.format(key), 'wb') as f:
#            pickle.dump(value, f)

def write_perturb_results(path, sample_iter, perturb_results):
    folder_path = os.path.join(path, 'sample_{}'.format(sample_iter))

    os.makedirs(folder_path)
    folder_path_f = os.path.join(folder_path, '{}.pickle')

    for key, value in perturb_results.items():
        with open(folder_path_f.format(key), 'wb') as f:
            pickle.dump(value, f)