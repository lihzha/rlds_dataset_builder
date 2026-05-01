# AgiBotWorld-Beta Dataset Structure

Source: https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta

~1M trajectories, 100 robots, 100+ scenarios, 200+ tasks, 87 atomic skills, ~48 TB total.

## On-disk layout (HuggingFace repo)

The repo ships **sharded `.tar` archives**, not loose files. Two sharding schemes are used:

```
observations/<task_id>/<ep_start>-<ep_end>.tar     # e.g. 786/927787-927793.tar
parameters/<ep_start>-<ep_end>.tar                  # e.g. 866961-940165.tar
proprio_stats/<ep_start>-<ep_end>.tar               # e.g. 866961-940165.tar
task_info/beta_task_info.zip
```

- **`observations/`** — tars are nested under `task_id`, each shard covers a small episode range (handful of episodes). Video + depth are large, so many small archives per task.
- **`parameters/` and `proprio_stats/`** — tars are flat at the top level (no `task_id` dir) and each shard covers a large cross-task episode range. Metadata is tiny, so many episodes bundle into one shard.
- **`task_info/`** — single zip containing all `task_<id>.json` annotation files.

## Logical layout (after extraction)

```
data/
├── task_info/
│   └── task_<id>.json
├── observations/<task_id>/<episode_id>/
│   ├── videos/*.mp4         # multi-camera RGB
│   └── depth/*.png          # depth frames
├── parameters/<task_id>/<episode_id>/
│   └── camera/...            # intrinsics/extrinsics
└── proprio_stats/<task_id>/<episode_id>/
    └── proprio_stats.h5
```

Organization is **task_id → episode_id**.

## Data formats

| Modality | Format |
|----------|--------|
| RGB video (multi-camera) | `.mp4` |
| Depth | `.png` per frame |
| Camera intrinsics/extrinsics | files under `camera/` |
| Proprioceptive state + action | `.h5` (HDF5) |
| Task / episode annotations | `.json` |

## `proprio_stats.h5` — fields (N = timesteps)

### State

| Field | Shape | Description |
|-------|-------|-------------|
| `state/joint/position` | `[N, 14]` | 7-DoF left + 7-DoF right arm (rad) |
| `state/end/position` | `[N, 2, 3]` | dual flange xyz (m) |
| `state/end/orientation` | `[N, 2, 4]` | dual flange quaternion (xyzw) |
| `state/effector/position` (gripper) | `[N, 2]` | left/right gripper opening (mm) |
| `state/effector/position` (dexhand) | `[N, 12]` | dexterous hand joint angles (rad) |
| `state/head/position` | `[N, 2]` | head yaw/pitch (rad) |
| `state/waist/position` | `[N, 2]` | waist pitch (rad) + lift (m) |
| `state/robot/position` | `[N, 3]` | base xy (z=0) (m) |
| `state/robot/orientation` | `[N, 4]` | base yaw quaternion (xyzw) |

Also `velocity`, `effort`, `wrench`, `force`, `current_value` under matching subgroups.

### Action

Mirrors the state hierarchy under `action/{effector,end,head,joint,robot,waist}/...`, plus `index` arrays marking the timesteps at which each action stream was issued.

Notable conventions:
- `action/effector/position` is 0/1 (open/close)
- `action/robot/velocity` is `[N, 2]` (forward velocity + yaw rate)

Top-level: `/timestamp [N]`.

## `task_<id>.json` — annotations

Per-episode entries:

```json
{
  "episode_id": 649078,
  "task_id": 327,
  "task_name": "Picking items in Supermarket",
  "init_scene_text": "The robot is in front of the fruit shelf...",
  "lable_info": {
    "action_config": [
      {"start_frame": 0, "end_frame": 435,
       "action_text": "Pick up onion from the shelf.",
       "skill": "Pick"}
    ],
    "key_frame": [
      {"start": 0, "end": 435, "comment": "Failure recovery"}
    ]
  }
}
```

- `action_config[]` — skill-segmented language labels (`skill` ∈ 87 atomic skills: Pick, Place, Peel, Tie, OpenJar, Sweep, …)
- `key_frame[]` — temporal annotations (e.g. failure/recovery)
- `init_scene_text` — initial scene description

## Hardware context

- Dual-arm, 7-DoF per arm
- End-effector: parallel gripper **or** 6-DoF dexterous hand per side
- 2-DoF head (pitch/yaw), 2-DoF waist (pitch + lift)
- Mobile base with odometry
- Visual-tactile sensing on some platforms
