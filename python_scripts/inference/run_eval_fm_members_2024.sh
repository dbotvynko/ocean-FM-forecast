#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=fm_members_2024
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-gpu=8
#SBATCH --array=0-3
#SBATCH --output=/Odyssey/private/d21botvy/job_%A_%a_fm_members_2024.log
#SBATCH --error=/Odyssey/private/d21botvy/%A_%a_fm_members_2024_error.txt
#
# OceanBench 2024: FM UNet forecasts with all 10 members saved, 48 Wednesday starts split over 4 array tasks
# (python_scripts/inference/eval_2024_oceanbench_fm_unet_members.py). Resumes with existing window files.
# Usage: sbatch [--begin=20:00] run_eval_fm_members_2024.sh <xp> <ckpt (abs path)> <tag> [<input .nc>]

export HOME=/Odyssey/private/d21botvy/
export TMPDIR=/tmp
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
cd /Odyssey/private/d21botvy/forecast/ocean-DDPMs

conda activate ddpm-env
export HYDRA_FULL_ERROR=1
srun python python_scripts/inference/eval_2024_oceanbench_fm_unet_members.py --xp "$1" --ckpt "$2" --tag "$3" \
    --part "$SLURM_ARRAY_TASK_ID" --nparts 4 ${4:+--input "$4"}
