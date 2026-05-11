#!/bin/bash

#SBATCH --job-name=mecka_part2    # Job name
#SBATCH --output=logs/%A_mecka_part2.out   # Output file
#SBATCH --error=logs/%A_mecka_part2.err    # Error file
#SBATCH --time=48:00:00            # Maximum runtime
#SBATCH -N 1
#SBATCH --gres=gpu:0            # Request 1 GPU
#SBATCH --ntasks-per-node=1          # 1 task per node
#SBATCH --cpus-per-task=15       # Match N_WORKERS
#SBATCH --mem=400G                    # Memory for shuffling phase
#SBATCH --partition=all              # Or specify GPU partition if needed

source /n/fs/robot-data/miniconda3/etc/profile.d/conda.sh

# Parameter configurations
conda activate rlds_zarr3
cd /n/fs/robot-data/rlds_multithread

# Build Part 2 (~5,202 episodes, indices 10403-15604, ~30K samples)
python3 scripts/build_mecka_part2.py
