#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=fm_coords_warmrestart
#SBATCH --ntasks-per-node=2
#SBATCH --mem=150G
#SBATCH --cpus-per-gpu=8
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_train_fm_unet_coords_warmrestart.log
#SBATCH --error=/Odyssey/private/d21botvy/%j_error.txt
#
# Trains config/xp/forecast_DDPM_UNet_1patch_coords_warmrestart.yaml. Same resources as
# the sin/cos coords launcher (2 GPUs on one node, one srun task per GPU for
# Lightning's SLURMEnvironment). Extra CLI args are passed to main.py as Hydra
# overrides. Hydra writes to its own outputs/<date>/<time>/ folder.

export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
cd /Odyssey/private/d21botvy/forecast/ocean-DDPMs

conda activate ddpm-env
export HYDRA_FULL_ERROR=1
srun python main.py xp=forecast_DDPM_UNet_1patch_coords_warmrestart "$@"
