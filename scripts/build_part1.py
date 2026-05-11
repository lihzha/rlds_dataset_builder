#!/usr/bin/env python3
"""Build script for Scale Dataset Part 1."""

import os
import sys

# Ensure the parent directory is in the Python path
sys.path.insert(0, '/n/fs/robot-data/rlds_multithread')

from scale_dataset.scale_dataset_dataset_builder import ScaleDatasetPart1

if __name__ == '__main__':
    print("Building Scale Dataset Part 1 (episodes 0-3794)...")

    # Create builder instance
    builder = ScaleDatasetPart1()

    # Download and prepare the dataset
    builder.download_and_prepare()

    print("Build completed successfully!")
