from collections.abc import Iterator
import json
import os
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import tensorflow_datasets as tfds

from droid_raw_dataset.conversion_utils import MultiThreadedDatasetBuilder


def _extract_video_frames(video_path: Path, num_frames: int, target_size=(84, 84)) -> np.ndarray:
    """Extract frames from MP4 video file.

    Args:
        video_path: Path to MP4 file
        num_frames: Expected number of frames to extract
        target_size: Target (height, width) for resizing frames

    Returns:
        Array of shape (num_frames, height, width, 3) with uint8 RGB frames
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize to target size
        frame_resized = cv2.resize(frame_rgb, target_size)
        frames.append(frame_resized)

    cap.release()

    frames_array = np.array(frames, dtype=np.uint8)

    # Verify we got expected number of frames
    if len(frames_array) != num_frames:
        print(f"WARNING: Expected {num_frames} frames from {video_path.name}, got {len(frames_array)}")
        # Pad or truncate if needed
        if len(frames_array) < num_frames:
            # Pad with last frame
            padding = np.repeat(frames_array[-1:], num_frames - len(frames_array), axis=0)
            frames_array = np.concatenate([frames_array, padding], axis=0)
        else:
            # Truncate
            frames_array = frames_array[:num_frames]

    return frames_array


def _generate_examples(paths) -> Iterator[tuple[str, Any]]:
    """Yields episodes for list of trajectory folder paths.

    Each path should point to a trajectory folder containing:
        - trajectory.h5: HDF5 with robot data
        - recordings/MP4/*.mp4: Video files for cameras
        - metadata_*.json: Metadata with task description

    HDF5 structure:
        /action/joint_velocity  shape=(T, 7)
        /action/gripper_position  shape=(T,)
        /observation/robot_state/cartesian_position  shape=(T, 6)  # [x, y, z, rx, ry, rz]
        /observation/robot_state/gripper_position  shape=(T,)
    """

    def _parse_trajectory_folder(traj_folder_path):
        """Parse a single trajectory folder.

        Args:
            traj_folder_path: Path to trajectory folder

        Yields:
            Tuple of (trajectory_key, trajectory_data)
        """
        traj_folder = Path(traj_folder_path)

        # Load HDF5 file
        h5_path = traj_folder / "trajectory.h5"
        if not h5_path.exists():
            raise FileNotFoundError(f"No trajectory.h5 found in {traj_folder}")

        # Load metadata JSON
        metadata_files = list(traj_folder.glob("metadata_*.json"))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata JSON found in {traj_folder}")

        with open(metadata_files[0]) as f:
            metadata = json.load(f)

        # Extract language instruction
        language_instruction = metadata.get("current_task", "")

        # Load video paths
        mp4_dir = traj_folder / "recordings" / "MP4"
        ext1_video = mp4_dir / "38872458.mp4"  # exterior_image_1_left
        wrist_video = mp4_dir / "10501775.mp4"  # wrist_image_left
        ext2_video = mp4_dir / "31177322.mp4"  # exterior_image_2_left

        # Verify video files exist
        for video_path in [ext1_video, wrist_video, ext2_video]:
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")

        # Load HDF5 data
        with h5py.File(h5_path, "r") as f:
            # Load actions: joint_velocity (7) + gripper_position (1) = 8D
            joint_velocity = np.array(f["action/joint_velocity"])  # (T, 7)
            action_gripper = np.array(f["action/gripper_position"])  # (T,)

            # Load observations
            cartesian_position = np.array(f["observation/robot_state/cartesian_position"])  # (T, 6)
            obs_gripper = np.array(f["observation/robot_state/gripper_position"])  # (T,)

            num_timesteps = len(joint_velocity)

            # Verify data alignment
            if not (
                len(action_gripper) == len(cartesian_position) == len(obs_gripper) == num_timesteps
            ):
                raise ValueError(
                    f"Data misalignment in {traj_folder.name}: "
                    f"joint_velocity={len(joint_velocity)}, action_gripper={len(action_gripper)}, "
                    f"cartesian_position={len(cartesian_position)}, obs_gripper={len(obs_gripper)}"
                )

        # Extract video frames
        print(f"Extracting frames from {traj_folder.name}...")
        ext1_frames = _extract_video_frames(ext1_video, num_timesteps)
        wrist_frames = _extract_video_frames(wrist_video, num_timesteps)
        ext2_frames = _extract_video_frames(ext2_video, num_timesteps)

        # Build episode steps
        episode = []
        for i in range(num_timesteps):
            # Construct state: cartesian_position (3) + euler_angle (3) + gripper (1) = 7D
            state = np.concatenate([
                cartesian_position[i, :3],  # position (x, y, z)
                cartesian_position[i, 3:6],  # euler angles (rx, ry, rz)
                obs_gripper[i:i+1],  # gripper position
            ]).astype(np.float32)

            # Construct action: joint_velocity (7) + gripper (1) = 8D
            action = np.concatenate([
                joint_velocity[i],  # (7,)
                action_gripper[i:i+1],  # (1,)
            ]).astype(np.float32)

            # Add step to episode
            episode.append({
                "observation": {
                    "exterior_image_1_left": ext1_frames[i],
                    "wrist_image_left": wrist_frames[i],
                    "exterior_image_2_left": ext2_frames[i],
                    "state": state,
                },
                "action": action,
                "discount": 1.0,
                "reward": float(i == (num_timesteps - 1)),
                "is_first": i == 0,
                "is_last": i == (num_timesteps - 1),
                "is_terminal": i == (num_timesteps - 1),
                "language_instruction": language_instruction,
            })

        # Create output data sample
        sample = {
            "steps": episode,
            "episode_metadata": {
                "file_path": str(h5_path),
                "folder_name": traj_folder.name,
            },
        }

        # Create unique key from folder name
        unique_key = traj_folder.name
        yield unique_key, sample

    # Parse examples from trajectory folder paths
    for traj_folder_path in paths:
        try:
            yield from _parse_trajectory_folder(traj_folder_path)
        except Exception as e:
            # Log the error
            print(f"WARNING: Skipping trajectory folder {traj_folder_path}: {type(e).__name__}: {e!s}")
            continue


class DroidRawDataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for planning dataset from HDF5 files."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    N_WORKERS = 10  # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 50  # number of paths converted & stored in memory before writing to disk
    PARSE_FCN = _generate_examples  # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "exterior_image_1_left": tfds.features.Image(
                                        shape=(84, 84, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Exterior camera 1 RGB observation (camera 38872458).",
                                    ),
                                    "wrist_image_left": tfds.features.Image(
                                        shape=(84, 84, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Wrist camera RGB observation (camera 10501775).",
                                    ),
                                    "exterior_image_2_left": tfds.features.Image(
                                        shape=(84, 84, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Exterior camera 2 RGB observation (camera 31177322).",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc="Robot state: [cartesian_pos (3), euler_angle (3), gripper (1)].",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(8,),
                                dtype=np.float32,
                                doc="Robot action: [joint_velocity (7), gripper (1)].",
                            ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Discount if provided, default to 1.",
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Reward if provided, 1 on final step for demos.",
                            ),
                            "is_first": tfds.features.Scalar(dtype=np.bool_, doc="True on first step of the episode."),
                            "is_last": tfds.features.Scalar(dtype=np.bool_, doc="True on last step of the episode."),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode if it is a terminal step, True for demos.",
                            ),
                            "language_instruction": tfds.features.Text(doc="Language Instruction."),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(doc="Path to the trajectory HDF5 file."),
                            "folder_name": tfds.features.Text(doc="Name of the trajectory folder."),
                        }
                    ),
                }
            )
        )

    def _split_paths(self):
        """Define filepaths for data splits."""
        # Get trajectory root directory from env variable, default to 2025-11-18/
        traj_root = Path(os.getenv("TRAJECTORY_ROOT_DIR", "/home/irom-lab/projects/openpi-cot/2025-11-18"))

        if not traj_root.exists():
            raise FileNotFoundError(
                f"Trajectory root directory not found: {traj_root}\n"
                f"Please set TRAJECTORY_ROOT_DIR environment variable or update the default path."
            )

        # Find all trajectory folders (each contains trajectory.h5)
        trajectory_folders = []
        for folder in sorted(traj_root.iterdir()):
            if folder.is_dir() and (folder / "trajectory.h5").exists():
                trajectory_folders.append(folder)

        if not trajectory_folders:
            raise FileNotFoundError(
                f"No trajectory folders found in {traj_root}\n"
                f"Expected folders with trajectory.h5 files."
            )

        print(f"Found {len(trajectory_folders)} trajectory folders in {traj_root}")

        return {
            "train": trajectory_folders,
        }
