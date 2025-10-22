# Sample Humanoid Robot Dataset

This dataset contains demonstrations of a humanoid robot performing various household tasks.

## Dataset Description

The dataset includes:
- 10 different tasks (washing bottles, etc.)
- 10 episodes total (1 episode per task)
- Multi-camera observations (8 RGB cameras + 1 depth camera)
- Proprioceptive state information (joint, effector, head, waist positions)
- Action trajectories (22-dimensional continuous actions)

## Camera Setup

The robot is equipped with:
- 1 head camera (480x640 RGB)
- 2 hand cameras (left/right, 480x640 RGB)
- 5 fisheye cameras (head center/left/right, back left/right, 748x960 RGB)
- 1 depth camera (head, 480x640)

## State and Action Space

- **State**: 20-dimensional vector containing joint positions, end-effector positions, head positions, and waist positions
- **Action**: 22-dimensional vector containing joint positions, end-effector positions, head positions, waist positions, and robot velocity

## Tasks

The dataset includes demonstrations for various household manipulation tasks such as:
- Washing bottles
- Pick and place operations
- Pouring tasks
- And more...

## Citation

If you use this dataset, please cite:

```bibtex
@misc{contributors2024agibotworldrepo,
  title={AgiBot World Colosseum},
  author={AgiBot World Colosseum contributors, Lihan Zha},
  howpublished={\url{https://github.com/OpenDriveLab/AgiBot-World}},
  year={2024}
}
```
