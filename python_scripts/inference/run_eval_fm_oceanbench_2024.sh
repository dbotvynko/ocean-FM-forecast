#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=eval_fm_ob2024
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-gpu=8
#SBATCH --output=/Odyssey/private/d21botvy/job_%j_eval_fm_ob2024.log
#SBATCH --error=/Odyssey/private/d21botvy/%j_error.txt
#
# OceanBench 2024: FM UNet forecasts for the 48 Wednesday starts of 2024 (sla_filtered NRT nadir input), RMSE vs the same
# L3 sla_filtered at obs points (python_scripts/inference/eval_2024_oceanbench_fm_unet_sla_filtered.py).
# sl-mee-br-206 allowed (2026-10-02: the only A100 node accepting jobs; the eval resumes with skip_existing).
# Usage: sbatch run_eval_sla_filtered_90d.sh <xp> <ckpt (abs path)> <tag>

export HOME=/Odyssey/private/d21botvy/
export TMPDIR=/tmp
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
cd /Odyssey/private/d21botvy/forecast/ocean-DDPMs

conda activate ddpm-env
export HYDRA_FULL_ERROR=1
srun python python_scripts/inference/eval_2024_oceanbench_fm_unet_sla_filtered.py --xp "$1" --ckpt "$2" --tag "$3"
