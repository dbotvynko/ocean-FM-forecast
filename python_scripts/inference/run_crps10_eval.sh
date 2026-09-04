#!/bin/bash
#SBATCH --partition=Odyssey
#SBATCH --job-name=crps10_fm_unet
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_crps10_fm_unet.log
#SBATCH --error=/Odyssey/private/d21botvy/%j_error.txt
# Selected partition
# Name for the job
# Resources asked
# %j for jobid
#
# No --nodelist pinned here (unlike the SST anomalies job this was copied
# from) -- add one back if you need a specific GPU node, e.g.:
#   #SBATCH --nodelist=sl-mee-br-209

export HOME=/Odyssey/private/d21botvy/
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
cd /Odyssey/private/d21botvy/forecast/ocean-DDPMs

conda activate ddpm-env
export HYDRA_FULL_ERROR=1
srun python python_scripts/inference/eval_nrt_2023_fm_unet_crps10.py
