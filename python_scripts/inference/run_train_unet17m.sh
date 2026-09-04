#!/bin/bash
#SBATCH --partition=Odyssey
#SBATCH --job-name=train_unet17m
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_train_unet17m.log
#SBATCH --error=/Odyssey/private/d21botvy/%j_error.txt
# Selected partition
# Name for the job
# Resources asked
# %j for jobid
#
# --gres=gpu:2 matches this xp's trainer.devices: 2. No --nodelist pinned
# by default -- add one back if you need a specific pair of GPU nodes,
# e.g.:
#   #SBATCH --nodelist=sl-mee-br-209

export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
cd /Odyssey/private/d21botvy/forecast/ocean-DDPMs

conda activate ddpm-env
export HYDRA_FULL_ERROR=1
srun python main.py xp=forecast_DDPM_UNet17M_1patch
