"""Episode parser for the AgiBot large dataset pipeline.

Extracted from agibot_large_dataset/agibot_large_dataset_dataset_builder.py.
The parser is path-driven: given an episode dir laid out as

    <base>/observations/<task_id>/<episode_id>/videos/*.mp4
    <base>/proprio_stats/<task_id>/<episode_id>/proprio_stats.h5
    <base>/task_info/task_<task_id>.json

it yields one (key, sample) pair per subskill, where `sample` matches the
schema in features.py.
"""

from pathlib import Path
import json
from typing import Iterator, Tuple, Any

import av
import cv2
import h5py
import numpy as np


def quaternion_to_euler(quat: np.ndarray) -> np.ndarray:
    """Batched (..., 4) xyzw quat -> (..., 3) extrinsic XYZ Euler (rad)."""
    quat = np.asarray(quat)
    assert quat.shape[-1] == 4
    qx, qy, qz, qw = np.moveaxis(quat, -1, 0)
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy**2 + qz**2)
    rz = np.arctan2(siny_cosp, cosy_cosp)
    sinp = np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0)
    ry = np.arcsin(sinp)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx**2 + qy**2)
    rx = np.arctan2(sinr_cosp, cosr_cosp)
    return np.stack([rx, ry, rz], axis=-1)


def _read_video_frames(video_path: Path) -> list:
    """Decode an mp4 to a list of 224x224 RGB frames using PyAV.

    PyAV with libdav1d is used instead of OpenCV because the AgiBot videos
    are encoded in AV1 codec, which OpenCV's bundled FFmpeg may not decode
    correctly on all systems.
    """
    if not video_path.exists():
        raise IOError(f"Video file not found: {video_path}")
    try:
        container = av.open(str(video_path))
    except av.error.InvalidDataError as e:
        raise IOError(f"Cannot open video file (possibly corrupted): {video_path}") from e

    frames = []
    for frame in container.decode(video=0):
        # Convert to RGB numpy array
        frame_rgb = frame.to_ndarray(format='rgb24')
        # Resize to 224x224
        frame_resized = cv2.resize(
            frame_rgb, (224, 224), interpolation=cv2.INTER_LINEAR
        )
        frames.append(frame_resized)
    container.close()
    return frames


def parse_episode(
    episode_path: Path, task_info_dir: Path, proprio_root: Path
) -> Iterator[Tuple[str, Any]]:
    """Parse one episode dir into a list of (unique_key, sample) per subskill.

    Args:
        episode_path: <base>/observations/<task_id>/<episode_id>
        task_info_dir: dir containing task_<id>.json files (shared scratch)
        proprio_root: dir containing <task_id>/<episode_id>/proprio_stats.h5
                      (shared scratch)

    Yields:
        (unique_key, sample_dict) tuples. unique_key is globally unique:
        "{task_id}_{episode_id}_{start_frame}".

    Raises:
        ValueError / IOError on malformed input — caller decides whether to
        log+skip or abort.
    """
    episode_path = Path(episode_path)
    task_id = episode_path.parent.name
    episode_id = episode_path.name

    task_json = task_info_dir / f"task_{task_id}.json"
    with open(task_json, "r") as f:
        task_data = json.load(f)

    episode_task_info = None
    for entry in task_data:
        if str(entry["episode_id"]) == episode_id:
            episode_task_info = entry
            break
    if episode_task_info is None:
        raise ValueError(f"Episode {episode_id} not found in task {task_id} JSON")

    action_configs = episode_task_info.get("label_info", {}).get("action_config", [])

    proprio_path = proprio_root / task_id / episode_id / "proprio_stats.h5"
    with h5py.File(proprio_path, "r") as f:
        gripper_state = np.array(f["state/effector/position"])  # (N, 2)
        ee_xyz = np.array(f["state/end/position"])  # (N, 2, 3)
        ee_xyzw = np.array(f["state/end/orientation"])  # (N, 2, 4)

        expected = {
            "gripper_state": (None, 2),
            "ee_xyz": (None, 2, 3),
            "ee_xyzw": (None, 2, 4),
        }
        for name, arr, exp in [
            ("gripper_state", gripper_state, expected["gripper_state"]),
            ("ee_xyz", ee_xyz, expected["ee_xyz"]),
            ("ee_xyzw", ee_xyzw, expected["ee_xyzw"]),
        ]:
            if len(arr.shape) != len(exp):
                raise ValueError(
                    f"Shape error: state {name} has shape {arr.shape}, "
                    f"expected {len(exp)} dims"
                )
            for i in range(1, len(exp)):
                if arr.shape[i] != exp[i]:
                    raise ValueError(
                        f"Shape error: state {name} has shape {arr.shape}, "
                        f"expected dim {i} to be {exp[i]}"
                    )

        gripper_action = np.array(f["action/effector/position"])
        ee_action_xyz = np.array(f["action/end/position"])
        ee_action_xyzw = np.array(f["action/end/orientation"])
        for name, arr, exp in [
            ("gripper_action", gripper_action, expected["gripper_state"]),
            ("ee_action_xyz", ee_action_xyz, expected["ee_xyz"]),
            ("ee_action_xyzw", ee_action_xyzw, expected["ee_xyzw"]),
        ]:
            if len(arr.shape) != len(exp):
                raise ValueError(
                    f"Shape error: action {name} has shape {arr.shape}, "
                    f"expected {len(exp)} dims"
                )
            for i in range(1, len(exp)):
                if arr.shape[i] != exp[i]:
                    raise ValueError(
                        f"Shape error: action {name} has shape {arr.shape}, "
                        f"expected dim {i} to be {exp[i]}"
                    )

        all_lengths = {
            len(gripper_state),
            len(ee_xyz),
            len(ee_xyzw),
            len(gripper_action),
            len(ee_action_xyz),
            len(ee_action_xyzw),
        }
        if len(all_lengths) > 1:
            raise ValueError(f"Proprio length mismatch: {all_lengths}")

        left_state = np.hstack(
            [
                ee_xyz[:, 0, :],
                quaternion_to_euler(ee_xyzw[:, 0, :]),
                gripper_state[:, 0:1],
            ]
        )
        right_state = np.hstack(
            [
                ee_xyz[:, 1, :],
                quaternion_to_euler(ee_xyzw[:, 1, :]),
                gripper_state[:, 1:2],
            ]
        )
        left_action = np.hstack(
            [
                ee_action_xyz[:, 0, :],
                quaternion_to_euler(ee_action_xyzw[:, 0, :]),
                gripper_action[:, 0:1],
            ]
        )
        right_action = np.hstack(
            [
                ee_action_xyz[:, 1, :],
                quaternion_to_euler(ee_action_xyzw[:, 1, :]),
                gripper_action[:, 1:2],
            ]
        )

    states = np.hstack([left_state, right_state]).astype(np.float32)
    actions = np.hstack([left_action, right_action]).astype(np.float32)

    video_dir = episode_path / "videos"
    video_files = {
        "head_color": video_dir / "head_color.mp4",
        "hand_left_color": video_dir / "hand_left_color.mp4",
        "hand_right_color": video_dir / "hand_right_color.mp4",
        "head_center_fisheye_color": video_dir / "head_center_fisheye_color.mp4",
    }
    video_frames = {k: _read_video_frames(p) for k, p in video_files.items()}

    counts = {k: len(v) for k, v in video_frames.items()}
    if len(set(counts.values())) > 1:
        raise ValueError(f"Frame count mismatch across cameras: {counts}")
    n_frames = next(iter(counts.values()))
    if n_frames != len(states) or n_frames != len(actions):
        raise ValueError(
            f"Frame/state/action length mismatch: frames={n_frames}, "
            f"states={len(states)}, actions={len(actions)}"
        )

    for action_config in action_configs:
        start_frame = action_config["start_frame"]
        end_frame = action_config["end_frame"]
        action_text = action_config.get("action_text", "").strip()
        skill_name = action_config.get("skill", "Unknown")
        language_instruction = (
            action_text if action_text else f"Episode {episode_id}_frame_{start_frame}"
        )

        steps = []
        sub_n = end_frame - start_frame
        for i in range(start_frame, end_frame):
            step_idx = i - start_frame
            steps.append(
                {
                    "observation": {
                        "head_image": video_frames["head_color"][i],
                        "hand_left_image": video_frames["hand_left_color"][i],
                        "hand_right_image": video_frames["hand_right_color"][i],
                        "head_center_fisheye_image": video_frames[
                            "head_center_fisheye_color"
                        ][i],
                        "state": states[i],
                    },
                    "action": actions[i],
                    "discount": 1.0,
                    "reward": float(step_idx == sub_n - 1),
                    "is_first": step_idx == 0,
                    "is_last": step_idx == sub_n - 1,
                    "is_terminal": step_idx == sub_n - 1,
                    "language_instruction": language_instruction,
                }
            )

        sample = {
            "steps": steps,
            "episode_metadata": {
                "file_path": str(episode_path),
                "task_id": int(task_id),
                "episode_id": int(episode_id),
                "original_episode_id": int(episode_id),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "subskill_name": skill_name,
            },
        }
        unique_key = f"{task_id}_{episode_id}_{start_frame}"
        yield unique_key, sample
