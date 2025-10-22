#!/bin/bash
#SBATCH --job-name=molmoact_build
#SBATCH --output=molmoact_build_%j.out
#SBATCH --error=molmoact_build_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# MolmoAct Dataset Build Job (Household + Tabletop)
# This script submits the dataset building process to SLURM

echo "=================================================="
echo "MolmoAct Dataset Build Job"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo ""

# Change to the script directory
cd /n/fs/robot-data/rlds_multithread

# Run the build script
bash build_molmoact_dataset.sh

# Capture exit status
EXIT_STATUS=$?

echo ""
echo "=================================================="
echo "Job Complete"
echo "=================================================="
echo "End time: $(date)"
echo "Exit status: $EXIT_STATUS"

# Print summary of the built dataset
if [ $EXIT_STATUS -eq 0 ]; then
    echo ""
    echo "Dataset successfully built!"
    echo ""
    echo "Summary:"

    # Activate conda to run Python commands
    source /n/fs/robot-data/miniconda3/etc/profile.d/conda.sh
    conda activate rlds

    python3 -c "
import tensorflow_datasets as tfds
try:
    builder = tfds.builder('molmoact_dataset')
    info = builder.info
    print(f'  Dataset name: {info.name}')
    print(f'  Version: {info.version}')
    print(f'  Splits: {list(info.splits.keys())}')
    if 'train' in info.splits:
        print(f'  Train episodes: {info.splits[\"train\"].num_examples:,}')
    print(f'  Features: {list(info.features.keys())}')
except Exception as e:
    print(f'  Could not load dataset info: {e}')
"
else
    echo ""
    echo "Build failed with exit status: $EXIT_STATUS"
    echo "Check the error log for details."
fi

exit $EXIT_STATUS
