#!/usr/bin/env python3
"""Build script for Mecka Dataset Part 1."""

import os
import sys

# Ensure the parent directory is in the Python path
sys.path.insert(0, '/n/fs/robot-data/rlds_multithread')

from mecka_dataset.mecka_dataset_dataset_builder import MeckaDatasetPart1

if __name__ == '__main__':
    print("Building Mecka Dataset Part 1 (episodes 0-10402)...")

    # Create builder instance
    builder = MeckaDatasetPart1()

    # Download and prepare the dataset
    builder.download_and_prepare()

    print("Build completed successfully!")
