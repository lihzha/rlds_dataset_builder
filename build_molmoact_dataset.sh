#!/bin/bash
#
# Build script for MolmoAct dataset (household + tabletop)
#
# This script converts the raw MolmoAct data (both household and tabletop)
# into TFDS format using multi-threaded processing.
#

set -e  # Exit on error

# Activate conda environment
source /n/fs/robot-data/miniconda3/etc/profile.d/conda.sh
conda activate rlds

# Set working directory
cd /n/fs/robot-data/rlds_multithread

echo "================================================================================"
echo "MolmoAct Dataset Builder (Household + Tabletop)"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  - Household data: /n/fs/robot-data/data/molmoact-dataset/data/molmoact_dataset_household"
echo "  - Tabletop data: /n/fs/robot-data/data/molmoact-dataset/data/molmoact_dataset_tabletop"
echo "  - Output: ~/tensorflow_datasets/molmoact_dataset"
echo "  - Workers: 10 parallel threads"
echo ""

# Load and display dataset statistics
echo "Loading dataset statistics..."
python3 -c "
import json

# Load household dataset info
with open('/n/fs/robot-data/data/molmoact-dataset/data/molmoact_dataset_household/train/meta/info.json', 'r') as f:
    household_info = json.load(f)

# Load tabletop dataset info
with open('/n/fs/robot-data/data/molmoact-dataset/data/molmoact_dataset_tabletop/train/meta/info.json', 'r') as f:
    tabletop_info = json.load(f)

print(f'Household Dataset:')
print(f'  - Episodes: {household_info[\"total_episodes\"]:,}')
print(f'  - Frames: {household_info[\"total_frames\"]:,}')
print(f'  - Tasks: {household_info[\"total_tasks\"]}')
print(f'')
print(f'Tabletop Dataset:')
print(f'  - Episodes: {tabletop_info[\"total_episodes\"]:,}')
print(f'  - Frames: {tabletop_info[\"total_frames\"]:,}')
print(f'  - Tasks: {tabletop_info[\"total_tasks\"]}')
print(f'')
print(f'Combined Total:')
print(f'  - Episodes: {household_info[\"total_episodes\"] + tabletop_info[\"total_episodes\"]:,}')
print(f'  - Frames: {household_info[\"total_frames\"] + tabletop_info[\"total_frames\"]:,}')
print(f'  - Tasks: {household_info[\"total_tasks\"] + tabletop_info[\"total_tasks\"]}')
"

echo ""
echo "================================================================================"
echo "Starting TFDS Build"
echo "================================================================================"
echo ""

# Run the build
tfds build molmoact_dataset --overwrite

echo ""
echo "================================================================================"
echo "Build Complete!"
echo "================================================================================"
echo ""
echo "Dataset location: ~/tensorflow_datasets/molmoact_dataset"
echo ""
echo "To verify the dataset:"
echo "  python3 -c 'import tensorflow_datasets as tfds; ds = tfds.load(\"molmoact_dataset\"); print(ds)'"
echo ""
