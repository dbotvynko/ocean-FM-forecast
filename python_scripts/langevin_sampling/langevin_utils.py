import torch
from tqdm import tqdm
import numpy as np
import os
import pickle
import scipy
import scipy.signal

def grad_log_p(x_original, obs, prior, beta, mask):
    x = x_original.clone()
    x = x.requires_grad_(True).cuda()
    x[~mask] = 0.0

    x = torch.nn.functional.pad(x, (0,0,0,0,15,7))
    obs = obs.clone().requires_grad_(True).cuda()

    phi_x = prior.forward_ae(x.unsqueeze(0)).squeeze(0)[14:14+7]
    x = x[14:14+7]
    E_x = torch.norm(x[mask] - phi_x[mask], p=2) ** 2

    msk_obs = obs.isfinite().cuda()
    y_cost = torch.norm(x[msk_obs] - obs[msk_obs], p=2) ** 2
    
    alpha = 0.00001
    Energy = torch.exp(-alpha*(E_x + y_cost))
    
    grad, = torch.autograd.grad(-beta * Energy, x, create_graph=True)
    del x
    del obs
    return grad.detach().cpu(), E_x.detach().cpu(), y_cost.detach().cpu(), Energy.detach().cpu()

def grad_log_p_old(x, obs, prior, alpha, mask):
    x = x.requires_grad_(True).cuda()

    obs = obs.clone().cuda()

    phi_x = prior.forward_ae(x.unsqueeze(0)).squeeze(0)
    E_x = torch.norm(x - phi_x, p=2) ** 2

    msk_x_metric = mask.cuda()
    msk_x_metric[:14] = 0.0
    msk_x_metric[-8:] = 0.0
    x_metric = torch.sqrt(torch.mean((x[msk_x_metric] - phi_x[msk_x_metric])**2))

    msk_obs_cost = obs.isfinite().cuda()
    msk_obs_cost[14:] = 0.0
    y_cost = torch.norm(x[msk_obs_cost] - obs[msk_obs_cost], p=2) ** 2

    msk_obs_metric = obs.isfinite().cuda()
    msk_obs_metric[:14] = 0.0
    msk_obs_metric[-8:] = 0.0
    y_metric = torch.sqrt(torch.mean((x[msk_obs_metric] - obs[msk_obs_metric])**2))

    
    Energy = torch.exp(-alpha*(E_x + y_cost))
    #Energy = torch.exp(-alpha*(E_x))
    
    grad, = torch.autograd.grad(-Energy, x, create_graph=True)
    grad = grad / alpha

    return grad.detach().cpu(), E_x.detach().cpu(), x_metric.detach().cpu(), y_cost.detach().cpu(), y_metric.detach().cpu(), Energy.detach().cpu()

def grad_log_p(x, obs, prior, alpha, mask):
    x = x.requires_grad_(True).cuda()

    obs = obs.clone().cuda()

    phi_x = prior.forward_ae(x.unsqueeze(0)).squeeze(0)
    E_x = torch.norm(x - phi_x, p=2) ** 2

    msk_x_metric = mask.cuda()
    msk_x_metric[:14] = 0.0
    msk_x_metric[-8:] = 0.0
    x_metric = torch.sqrt(torch.mean((x[msk_x_metric] - phi_x[msk_x_metric])**2))

    msk_obs_cost = obs.isfinite().cuda()
    msk_obs_cost[14:] = 0.0
    y_cost = torch.norm(x[msk_obs_cost] - obs[msk_obs_cost], p=2) ** 2

    msk_obs_metric = obs.isfinite().cuda()
    msk_obs_metric[:14] = 0.0
    msk_obs_metric[-8:] = 0.0
    y_metric = torch.sqrt(torch.mean((x[msk_obs_metric] - obs[msk_obs_metric])**2))

    
    Energy = -(E_x + y_cost)
    
    grad, = torch.autograd.grad(Energy, x, create_graph=True)

    return grad.detach().cpu(), E_x.detach().cpu(), x_metric.detach().cpu(), y_cost.detach().cpu(), y_metric.detach().cpu(), Energy.detach().cpu()



def langevin_sampling(X_0, obs, prior, array_mask, num_samples=15, step_size=0.0001, alpha=1.0, noise_scaling=0.1, return_samples=False, return_last_sample=False):
    samples = []
    x_costs = []
    x_metrics = []
    y_costs = []
    y_metrics = []
    energies = []
    #X = obs.nan_to_num()
    X = X_0
    
    for _ in tqdm(range(num_samples), desc='langevin sampling:'):
        noise = np.sqrt(2 * step_size * noise_scaling) * np.random.normal(0.0, 1.0, X_0.shape)
        #noise = torch.zeros_like(X)
        grad, x_cost, x_metric, y_cost, y_metric, energy = grad_log_p(X, obs, prior, alpha, array_mask.cuda())
        X = X + step_size * grad + torch.Tensor(noise)
        
        x_costs.append(x_cost)
        x_metrics.append(x_metric)
        y_costs.append(y_cost)
        y_metrics.append(y_metric)
        energies.append(energy)
        if return_samples:
            samples.append(X.clone().detach().cpu().numpy())
    
    langevin_results = dict()
    if return_samples:
        langevin_results['samples'] = np.array(samples)
    langevin_results['energies'] = np.array(energies)
    langevin_results['x_costs'] = np.array(x_costs)
    langevin_results['x_metrics'] = np.array(x_metrics)
    langevin_results['y_costs'] = np.array(y_costs)
    langevin_results['y_metrics'] = np.array(y_metrics)

    if return_last_sample:
        langevin_results['last_sample'] = X.clone().detach().cpu().numpy()

    return langevin_results

def write_langevin_results(path, langevin_results):
    folder_name = 'sampling'
    folder_path = os.path.join(path, '{}_0'.format(folder_name))
    counter = 0
    while os.path.exists(folder_path):
        counter+=1
        folder_path = os.path.join(path, '{}_{}'.format(folder_name, counter))
    os.makedirs(folder_path)
    folder_path_f = os.path.join(folder_path, '{}.pickle')

    for key, value in langevin_results.items():
        with open(folder_path_f.format(key), 'wb') as f:
            pickle.dump(value, f)

def crps_image_iterative(pred_samples: torch.Tensor, y_true: torch.Tensor, device="cuda") -> np.ndarray:
    """
    Compute the Continuous Ranked Probability Score (CRPS) for image outputs iteratively over t to reduce memory usage.

    Args:
        pred_samples (torch.Tensor): Predicted samples from the model (num_samples, t, x, y) (stored in CPU).
        y_true (torch.Tensor): Ground truth image (t, x, y) (stored in CPU).
        device (str): Device to use for computation ("cuda" or "cpu").

    Returns:
        np.ndarray: CRPS score per pixel (t, x, y), with NaNs where y_true is NaN.
    """
    num_samples, t, x, y = pred_samples.shape

    # Create output CRPS array in CPU memory
    crps_result = np.full((t, x, y), np.nan, dtype=np.float32)

    for i in range(t):  # Iterate over time dimension
        y_true_slice = y_true[i]  # Shape: (x, y)

        # Skip iteration if y_true_slice is completely NaN
        if torch.all(y_true_slice.isnan()):
            continue

        # Move only this time slice to GPU if using CUDA
        y_true_slice = y_true_slice.to(device)

        # Create valid mask (CPU)
        #valid_mask = y_true_slice.isfinite()

        # Fetch only relevant slice from pred_samples and move it to GPU
        pred_samples_slice = pred_samples[:, i].to(device)  # Shape: (num_samples, x, y)

        # Expand y_true_slice to match (num_samples, x, y)
        y_true_expanded = y_true_slice.unsqueeze(0).expand(num_samples, -1, -1)

        # Compute empirical CDF for this time step
        indicator = (pred_samples_slice < y_true_expanded).float()
        empirical_cdf = indicator.mean(dim=0)  # Average over num_samples

        # Compute CRPS
        crps_slice = torch.mean((empirical_cdf - (y_true_expanded > pred_samples_slice).float())**2, dim=0)

        # Convert to NumPy and store results, preserving NaNs
        crps_numpy = crps_slice.cpu().numpy()
        #crps_numpy[~valid_mask.cpu().numpy()] = np.nan
        crps_result[i] = crps_numpy

        # Free up memory
        del y_true_slice, pred_samples_slice, y_true_expanded, indicator, empirical_cdf, crps_slice
        torch.cuda.empty_cache()

    return crps_result  # Shape (t, x, y)

def crps_image_iterative_cpu(pred_samples: torch.Tensor, y_true: torch.Tensor, device="cuda") -> np.ndarray:
    """
    Compute the Continuous Ranked Probability Score (CRPS) for image outputs iteratively over t to reduce memory usage.

    Args:
        pred_samples (torch.Tensor): Predicted samples from the model (num_samples, t, x, y) (stored in CPU).
        y_true (torch.Tensor): Ground truth image (t, x, y) (stored in CPU).
        device (str): Device to use for computation ("cuda" or "cpu").

    Returns:
        np.ndarray: CRPS score per pixel (t, x, y), with NaNs where y_true is NaN.
    """
    num_samples, t, x, y = pred_samples.shape

    # Create output CRPS array in CPU memory
    crps_result = np.full((t, x, y), np.nan, dtype=np.float32)

    for i in range(t):  # Iterate over time dimension
        y_true_slice = y_true[i]  # Shape: (x, y)

        # Skip iteration if y_true_slice is completely NaN
        if torch.all(y_true_slice.isnan()):
            continue

        # Move only this time slice to GPU if using CUDA
        y_true_slice = y_true_slice

        # Create valid mask (CPU)
        valid_mask = y_true_slice.isfinite()

        # Fetch only relevant slice from pred_samples and move it to GPU
        pred_samples_slice = pred_samples[:, i]  # Shape: (num_samples, x, y)

        # Expand y_true_slice to match (num_samples, x, y)
        y_true_expanded = y_true_slice.unsqueeze(0).expand(num_samples, -1, -1)

        # Compute empirical CDF for this time step
        indicator = (pred_samples_slice < y_true_expanded).float()
        empirical_cdf = indicator.mean(dim=0)  # Average over num_samples

        # Compute CRPS
        crps_slice = torch.mean((empirical_cdf - (y_true_expanded > pred_samples_slice).float())**2, dim=0)

        # Convert to NumPy and store results, preserving NaNs
        crps_numpy = crps_slice.cpu().numpy()
        crps_numpy[~valid_mask.cpu().numpy()] = np.nan
        crps_result[i] = crps_numpy

        # Free up memory
        del y_true_slice, pred_samples_slice, y_true_expanded, indicator, empirical_cdf, crps_slice
        torch.cuda.empty_cache()

    return crps_result  # Shape (t, x, y)


def mvh_corr(mat1, mat2):
    return np.mean(
            [
                np.mean(scipy.stats.pearsonr(mat1[14], mat2[14])[0]),
                np.mean(scipy.stats.pearsonr(mat1[14], mat2[14], axis=1)[0])
            ]
        )

def cross_corr(mat1, mat2, leadtime=None):
    cross_corrs = []
    if leadtime is None:
        for i in range(mat1.shape[0]):
            cross_corrs.append(scipy.signal.correlate2d(mat1[i], mat2[i]))
    else:
        return scipy.signal.correlate2d(mat1[leadtime], mat2[leadtime]).item()
    return np.mean(cross_corrs)