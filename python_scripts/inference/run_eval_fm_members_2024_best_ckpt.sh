#!/bin/bash
#SBATCH --partition=Odyssey_GPU
#SBATCH --job-name=fm_members_2024_best
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-gpu=8
#SBATCH --array=0-3
#SBATCH --output=/Odyssey/private/d21botvy/job_%A_%a_fm_members_2024.log
#SBATCH --error=/Odyssey/private/d21botvy/%A_%a_fm_members_2024_error.txt
#
# OceanBench 2024: FM UNet forecasts with all 10 members saved, 48 Wednesday starts split over 4 array tasks
# (python_scripts/inference/eval_2024_oceanbench_fm_unet_members.py). Resumes with existing window files.
# Same as run_eval_fm_members_2024.sh, but on the best checkpoint of the latest run of <xp>, resolved at start
# time (so it can be queued with --dependency on the training job).
# Usage: sbatch [--dependency=afterany:<train job>] run_eval_fm_members_2024_best_ckpt.sh <xp> <tag> [<input .nc>]

export HOME=/Odyssey/private/d21botvy/
export TMPDIR=/tmp
source "/Odyssey/private/d21botvy/miniconda3/etc/profile.d/conda.sh"
cd /Odyssey/private/d21botvy/forecast/ocean-DDPMs

conda activate ddpm-env
export HYDRA_FULL_ERROR=1
RUN_DIR=$(ls -d outputs/*/*/"$1" | sort | tail -1)
CKPT=$PWD/$(ls "$RUN_DIR"/checkpoints/val_loss=*.ckpt | sort | head -1)
echo "Checkpoint used: $CKPT"
srun python python_scripts/inference/eval_2024_oceanbench_fm_unet_members.py --xp "$1" --ckpt "$CKPT" --tag "$2" \
    --part "$SLURM_ARRAY_TASK_ID" --nparts 4 ${3:+--input "$3"}
