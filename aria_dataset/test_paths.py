#!/usr/bin/env python3
"""Test script to verify aria dataset path loading."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aria_dataset.aria_dataset_dataset_builder import AriaDataset

# Create dataset builder
builder = AriaDataset()

# Test _split_paths
print("Testing _split_paths()...")
paths = builder._split_paths()

print(f"\nSplits: {list(paths.keys())}")
for split_name, split_paths in paths.items():
    print(f"\n{split_name}:")
    print(f"  Total paths: {len(split_paths)}")
    if len(split_paths) > 0:
        print(f"  First path: {split_paths[0]}")
        print(f"  Last path: {split_paths[-1]}")

        # Verify first path
        first_path = Path(split_paths[0])
        print(f"  First path exists: {first_path.exists()}")
        print(f"  First path is dir: {first_path.is_dir()}")
        print(f"  First path has zarr.json: {(first_path / 'zarr.json').exists()}")

print("\n✓ Path loading test passed!")
