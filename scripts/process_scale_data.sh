#!/bin/bash

#SBATCH --job-name=scale_data    # Job name
#SBATCH --output=logs/%A_dpdata.out   # Output file
#SBATCH --error=logs/%A_dpdata.err    # Error file
#SBATCH --time=48:00:00            # Maximum runtime
#SBATCH -N 1
#SBATCH --gres=gpu:0            # Request 1 GPU
#SBATCH --ntasks-per-node=1          # 1 task per node
#SBATCH --cpus-per-task=15       # Reduced to match N_WORKERS (was 30)
#SBATCH --mem=400G                    # Increased memory for shuffling phase (was 300G)
#SBATCH --partition=all              # Or specify GPU partition if needed

source /n/fs/robot-data/miniconda3/etc/profile.d/conda.sh

# Parameter configurations
conda activate rlds_zarr3
cd /n/fs/robot-data/rlds_multithread/scale_dataset

tfds build --overwrite