# Planning Dataset

This dataset contains robot demonstrations for planning tasks with visual observations and proprioceptive state information, loaded from HDF5 files.

## Dataset Description

The dataset includes:
- Robot demonstrations with dual camera views (base and wrist)
- Proprioceptive state information (arm position, orientation, gripper state)
- Action trajectories (10-dimensional continuous actions)
- Object state information (base pose, cube positions and orientations)

## HDF5 Data Format

Each HDF5 file contains multiple demonstrations under the `/data` group:

```
/data/
  demo_0/
    actions          shape=(T, 10) dtype=float64
    obs/
      arm_pos        shape=(T, 3) dtype=float64
      arm_quat       shape=(T, 4) dtype=float64
      base_image     shape=(T, 84, 84, 3) dtype=uint8
      wrist_image    shape=(T, 84, 84, 3) dtype=uint8
      gripper_pos    shape=(T, 1) dtype=float64
      base_pose      shape=(T, 3) dtype=float64
      cube1_pos      shape=(T, 3) dtype=float64
      cube1_quat     shape=(T, 4) dtype=float64
      cube2_pos      shape=(T, 3) dtype=float64
      cube2_quat     shape=(T, 4) dtype=float64
      cube3_pos      shape=(T, 3) dtype=float64
      cube3_quat     shape=(T, 4) dtype=float64
  demo_1/
    ...
  demo_N/
    ...
```

Where `T` is the number of timesteps in each demonstration.

### Required Fields

The following fields are **required** for each demo:
- `actions`: 10-dimensional action vector at each timestep
- `obs/arm_pos`: End-effector position [x, y, z]
- `obs/arm_quat`: End-effector orientation as quaternion [x, y, z, w]
- `obs/gripper_pos`: Gripper position (1D)
- `obs/base_image`: Base camera RGB image (84×84×3)
- `obs/wrist_image`: Wrist camera RGB image (84×84×3)

### Optional Fields

The following fields are **optional** and loaded if present:
- `obs/base_pose`: Robot base position
- `obs/cube{1,2,3}_pos`: Position of cube objects
- `obs/cube{1,2,3}_quat`: Orientation of cube objects

### Observations

- **base_image**: Base camera RGB image (84×84×3, uint8)
- **wrist_image**: Wrist camera RGB image (84×84×3, uint8)
- **state**: 8-dimensional vector containing:
  - arm_pos (3): End-effector position [x, y, z]
  - arm_quat (4): End-effector orientation as quaternion [x, y, z, w]
  - gripper_pos (1): Gripper position

### Actions

- **action**: 10-dimensional continuous action vector

## State and Action Space

- **State**: 8-dimensional vector [arm_pos(3), arm_quat(4), gripper_pos(1)]
- **Action**: 10-dimensional vector (format depends on your robot/task)

## Usage

### 1. Update Data Path

First, update the HDF5 file path in `planning_dataset_dataset_builder.py` at line 224:

```python
base_path = Path("/path/to/your/hdf5/files")
```

Point this to the directory containing your `.hdf5` or `.h5` files.

### 2. Install the Package

```bash
cd /path/to/rlds_dataset_builder
pip install -e .
```

### 3. Build the Dataset

```bash
cd planning_dataset
export CUDA_VISIBLE_DEVICES=  # Disable GPU for data processing
tfds build --overwrite
```

The dataset will be built to `~/tensorflow_datasets/planning_dataset/`.

### 4. Visualize the Dataset

```bash
python3 ../visualize_dataset.py planning_dataset
```

## Configuration

You can adjust the following parameters in `planning_dataset_dataset_builder.py`:

- **N_WORKERS**: Number of parallel workers (default: 10)
- **MAX_PATHS_IN_MEMORY**: Number of HDF5 files processed in memory before writing (default: 50)

## Expected Directory Structure

```
/path/to/your/hdf5/files/
├── data_file_1.hdf5  (contains demo_0, demo_1, ..., demo_N)
├── data_file_2.hdf5  (contains demo_0, demo_1, ..., demo_M)
└── ...
```

Each HDF5 file can contain an arbitrary number of demonstrations. The builder will iterate through all demos in all HDF5 files.

## Output Format

The converted RLDS dataset will have the following structure:

- **steps**: Sequence of timesteps containing:
  - **observation**: Dict with base_image, wrist_image, and state
  - **action**: 10-dimensional action vector
  - **reward**: 1.0 on the last step, 0.0 otherwise
  - **discount**: 1.0 for all steps
  - **is_first**: True on the first step
  - **is_last**: True on the last step
  - **is_terminal**: True on the last step
  - **language_instruction**: Text instruction (e.g., "demo_0")

- **episode_metadata**: Dict with:
  - **file_path**: Path to the original HDF5 file
  - **demo_name**: Name of the demo (e.g., "demo_0")

## Citation

If you use this dataset, please cite:

```bibtex
@misc{planning_dataset2024,
  title={Planning Dataset for Robot Manipulation},
  author={Planning Dataset Contributors},
  year={2024}
}
```
