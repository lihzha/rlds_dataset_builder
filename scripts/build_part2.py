#!/usr/bin/env python3
"""Build script for Scale Dataset Part 2."""

import os
import sys

# Ensure the parent directory is in the Python path
sys.path.insert(0, '/n/fs/robot-data/rlds_multithread')

from scale_dataset.scale_dataset_dataset_builder import ScaleDatasetPart2

if __name__ == '__main__':
    print("Building Scale Dataset Part 2 (episodes 3795-end)...")

    # Create builder instance
    builder = ScaleDatasetPart2()

    # Download and prepare the dataset
    builder.download_and_prepare()

    print("Build completed successfully!")
