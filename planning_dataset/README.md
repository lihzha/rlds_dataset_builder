# Planning Dataset

This dataset contains robot demonstrations for planning tasks with visual observations and proprioceptive state information.

## Dataset Description

The dataset includes:
- Robot demonstrations with dual camera views (base and wrist)
- Proprioceptive state information (arm position, orientation, gripper state)
- Action trajectories (8-dimensional continuous actions)
- Timestamped observations and actions

## Data Format

Each demonstration contains:
- `data.pkl`: Pickle file containing timestamps, observations, and actions
- `base_image.mp4`: Base camera video (360x640 resolution)
- `wrist_image.mp4`: Wrist camera video (480x640 resolution)

### Observations

- **base_image**: Base camera RGB image (360x640x3)
- **wrist_image**: Wrist camera RGB image (480x640x3)
- **state**: 8-dimensional vector containing:
  - arm_pos (3): End-effector position [x, y, z]
  - arm_quat (4): End-effector orientation as quaternion [x, y, z, w]
  - gripper_pos (1): Gripper position

### Actions

- **action**: 8-dimensional vector containing:
  - arm_pos (3): Target end-effector position [x, y, z]
  - arm_quat (4): Target end-effector orientation as quaternion [x, y, z, w]
  - gripper_pos (1): Target gripper position

## State and Action Space

- **State**: 8-dimensional vector [arm_pos(3), arm_quat(4), gripper_pos(1)]
- **Action**: 8-dimensional vector [arm_pos(3), arm_quat(4), gripper_pos(1)]

## Usage

To build the dataset:

```bash
cd planning_dataset
tfds build --overwrite
```

To visualize the dataset:

```bash
python3 ../visualize_dataset.py planning_dataset
```

## Dataset Structure

```
demos/
├── demo_0/
│   ├── data.pkl
│   ├── base_image.mp4
│   └── wrist_image.mp4
├── demo_1/
│   ├── data.pkl
│   ├── base_image.mp4
│   └── wrist_image.mp4
└── ...
```

## Citation

If you use this dataset, please cite:

```bibtex
@misc{planning_dataset2024,
  title={Planning Dataset for Robot Manipulation},
  author={Planning Dataset Contributors},
  year={2024}
}
```
