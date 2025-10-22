#!/bin/bash
#
# Build script for AgiBotWorld-Beta large dataset with subskill splitting
#
# This script converts the raw AgiBotWorld-Beta data into TFDS format
# with episodes split by subskills based on action_config.
#

set -e  # Exit on error

# Activate conda environment
source /n/fs/robot-data/miniconda3/etc/profile.d/conda.sh
conda activate rlds

# Set working directory
cd /n/fs/robot-data/rlds_multithread

# Check if tracking file exists
if [ ! -f "processed_episodes.json" ]; then
    echo "ERROR: Tracking file not found!"
    echo "Please run scan_and_track_episodes.py first:"
    echo "  python3 scan_and_track_episodes.py"
    exit 1
fi


echo ""
echo "================================================================================"
echo "Starting TFDS Build"
echo "================================================================================"
echo ""

# Run the build
tfds build agibot_large_dataset --overwrite

echo ""
echo "================================================================================"
echo "Build Complete!"
echo "================================================================================"
echo ""
echo "Dataset location: ~/tensorflow_datasets/agibot_large_dataset"
echo ""
echo "To verify the dataset:"
echo "  python3 -c 'import tensorflow_datasets as tfds; ds = tfds.load(\"agibot_large_dataset\"); print(ds)'"
echo ""
