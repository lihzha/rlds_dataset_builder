#!/usr/bin/env python3
"""Diagnostic script to identify zarr loading issues."""

import sys
from pathlib import Path
import zarr
import traceback

sys.path.insert(0, '/n/fs/robot-data/rlds_multithread')

print("="*80)
print("ZARR DIAGNOSTIC SCRIPT")
print("="*80)
print(f"Python version: {sys.version}")
print(f"Zarr version: {zarr.__version__}")
print()

# Test path from error message
test_path = Path("/n/fs/robot-data/EgoVerse/data/aria_fold_clothes/2025-09-20-18-44-29-000000")

print(f"Testing path: {test_path}")
print(f"  Exists: {test_path.exists()}")
print(f"  Is directory: {test_path.is_dir() if test_path.exists() else 'N/A'}")
print(f"  Has zarr.json: {(test_path / 'zarr.json').exists() if test_path.exists() else 'N/A'}")
print()

if not test_path.exists():
    print("ERROR: Path does not exist!")
    sys.exit(1)

# Test different opening methods
methods = [
    ("zarr.open(Path)", lambda: zarr.open(test_path, mode='r')),
    ("zarr.open(str)", lambda: zarr.open(str(test_path), mode='r')),
    ("zarr.open_group(Path)", lambda: zarr.open_group(test_path, mode='r')),
    ("zarr.open_group(str)", lambda: zarr.open_group(str(test_path), mode='r')),
]

for method_name, method_func in methods:
    print("="*80)
    print(f"Testing: {method_name}")
    print("-"*80)
    try:
        store = method_func()
        print(f"✓ Successfully opened store")
        print(f"  Store type: {type(store)}")
        print(f"  Store keys: {list(store.keys())[:5]}...")  # Show first 5 keys

        # Try to access attributes
        try:
            attrs = store.attrs.asdict()
            print(f"  Attributes: {list(attrs.keys())}")
        except Exception as e:
            print(f"  ✗ Failed to access attrs: {e}")

        # Try to access a data array
        try:
            images = store['images.front_1']
            print(f"  Images array: shape={images.shape}, dtype={images.dtype}")

            # Try to load actual data
            first_frame = images[0]
            print(f"  ✓ Successfully loaded first frame: type={type(first_frame)}")
        except Exception as e:
            print(f"  ✗ Failed to access images array: {e}")
            traceback.print_exc()

    except Exception as e:
        print(f"✗ Failed to open: {type(e).__name__}: {e}")
        traceback.print_exc()
    print()

# Now test the actual tfds builder's _parse_example function
print("="*80)
print("Testing actual _parse_example function from builder")
print("="*80)

try:
    from aria_dataset.aria_dataset_dataset_builder import _generate_examples

    print("Calling _generate_examples with one path...")
    paths = [str(test_path)]

    results = list(_generate_examples(paths))

    if results:
        print(f"✓ Successfully generated {len(results)} examples")
        for key, data in results[:2]:  # Show first 2
            print(f"  Example key: {key}")
            print(f"  Has steps: {'steps' in data}")
            print(f"  Has metadata: {'episode_metadata' in data}")
    else:
        print("✗ No results generated (but no error either)")

except Exception as e:
    print(f"✗ Error calling _generate_examples: {type(e).__name__}: {e}")
    traceback.print_exc()

print()
print("="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
