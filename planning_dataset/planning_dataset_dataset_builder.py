from collections.abc import Iterator
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import tensorflow_datasets as tfds

from planning_dataset.conversion_utils import MultiThreadedDatasetBuilder


def _quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [w, x, y, z] -> axis-angle vector (so(3))."""
    # Ensure q[0] >= 0 to keep the short path
    if q[0] < 0:
        q = -q
    w, x, y, z = q
    w = np.clip(w, -1.0, 1.0)
    half = np.arccos(w)
    theta = 2.0 * half
    s = np.sin(half)
    if theta < 1e-12 or s < 1e-12:
        return np.zeros(3, dtype=q.dtype)
    axis = np.array([x, y, z]) / s
    return axis * theta


def _generate_examples(paths) -> Iterator[tuple[str, Any]]:
    """Yields episodes for list of HDF5 file paths.

    Each path should point to an HDF5 file containing multiple demos.
    HDF5 structure:
        /data/demo_0/actions  shape=(T, 10)
        /data/demo_0/obs/arm_pos  shape=(T, 3)
        /data/demo_0/obs/arm_quat  shape=(T, 4)
        /data/demo_0/obs/base_image  shape=(T, 84, 84, 3)
        /data/demo_0/obs/wrist_image  shape=(T, 84, 84, 3)
        /data/demo_0/obs/gripper_pos  shape=(T, 1)
        /data/demo_0/obs/base_pose  shape=(T, 3)
        /data/demo_0/obs/cube{1,2,3}_pos  shape=(T, 3)
        /data/demo_0/obs/cube{1,2,3}_quat  shape=(T, 4)
    """

    def _parse_hdf5_file(hdf5_path):
        """Parse all demos from a single HDF5 file.

        Args:
            hdf5_path: Path to HDF5 file

        Yields:
            Tuple of (demo_key, demo_data) for each demo in the file
        """
        hdf5_path = Path(hdf5_path)

        with h5py.File(hdf5_path, "r") as f:
            # Get all demo groups (e.g., demo_0, demo_1, ...)
            if "data" not in f:
                raise ValueError(f"HDF5 file {hdf5_path} missing 'data' group")

            data_group = f["data"]
            demo_names = sorted(data_group.keys())

            for demo_name in demo_names:
                demo_group = data_group[demo_name]

                # Load actions
                actions = np.array(demo_group["actions"])  # shape (T, 10)

                # Load observations
                obs_group = demo_group["obs"]
                arm_pos = np.array(obs_group["arm_pos"])  # shape (T, 3)
                arm_quat = np.array(obs_group["arm_quat"])  # shape (T, 4)
                gripper_pos = np.array(obs_group["gripper_pos"])  # shape (T, 1)
                base_image = np.array(obs_group["base_image"])  # shape (T, 84, 84, 3)
                wrist_image = np.array(obs_group["wrist_image"])  # shape (T, 84, 84, 3)

                # Optional: Load object poses (cubes, base_pose) for metadata
                base_pose = np.array(obs_group["base_pose"]) if "base_pose" in obs_group else None
                cube1_pos = np.array(obs_group["cube1_pos"]) if "cube1_pos" in obs_group else None
                cube1_quat = np.array(obs_group["cube1_quat"]) if "cube1_quat" in obs_group else None
                cube2_pos = np.array(obs_group["cube2_pos"]) if "cube2_pos" in obs_group else None
                cube2_quat = np.array(obs_group["cube2_quat"]) if "cube2_quat" in obs_group else None
                cube3_pos = np.array(obs_group["cube3_pos"]) if "cube3_pos" in obs_group else None
                cube3_quat = np.array(obs_group["cube3_quat"]) if "cube3_quat" in obs_group else None

                language_instructions = demo_group["language"]

                # Verify data alignment
                num_timesteps = len(actions)
                if not (
                    len(arm_pos)
                    == len(arm_quat)
                    == len(gripper_pos)
                    == len(base_image)
                    == len(wrist_image)
                    == num_timesteps
                ):
                    raise ValueError(
                        f"Data misalignment in {hdf5_path}/{demo_name}: "
                        f"actions={len(actions)}, arm_pos={len(arm_pos)}, "
                        f"arm_quat={len(arm_quat)}, gripper_pos={len(gripper_pos)}, "
                        f"base_image={len(base_image)}, wrist_image={len(wrist_image)}"
                    )

                # Build episode steps
                episode = []
                for i in range(num_timesteps):
                    # Construct state: arm_pos (3) + arm_quat (4) + gripper_pos (1) = 8
                    state = np.concatenate(
                        [
                            base_pose[i],  # (3,)
                            arm_pos[i],  # (3,)
                            arm_quat[i],  # (4,)
                            gripper_pos[i],  # (1,)
                        ]
                    ).astype(np.float32)

                    # Actions are 10-dimensional, we'll store them as-is
                    action_vector = actions[i].astype(np.float32)
                    if len(action_vector) == 11:
                        # turn into axis angle
                        action_vector = np.concatenate(
                            [action_vector[:6], _quat_to_axis_angle(action_vector[6:10]), action_vector[10:]], axis=-1
                        )

                    # Add step to episode
                    episode.append(
                        {
                            "observation": {
                                "base_image": base_image[i].astype(np.uint8),
                                "wrist_image": wrist_image[i].astype(np.uint8),
                                "state": state,
                            },
                            "action": action_vector,
                            "discount": 1.0,
                            "reward": float(i == (num_timesteps - 1)),
                            "is_first": i == 0,
                            "is_last": i == (num_timesteps - 1),
                            "is_terminal": i == (num_timesteps - 1),
                            # "language_instruction": "Pick the red cube and place it in the +x direction by 0.5m",
                            "language_instruction": np.array(language_instructions).item().decode(),
                        }
                    )

                # Create output data sample
                sample = {
                    "steps": episode,
                    "episode_metadata": {
                        "file_path": str(hdf5_path),
                        "demo_name": demo_name,
                    },
                }

                # Create unique key combining file name and demo name
                unique_key = f"{hdf5_path.stem}_{demo_name}"
                yield unique_key, sample

    # Parse examples from HDF5 file paths
    for hdf5_path in paths:
        try:
            # Each HDF5 file may contain multiple demos
            for unique_key, episode_data in _parse_hdf5_file(hdf5_path):
                yield unique_key, episode_data
        except Exception as e:
            # Log the error
            print(f"WARNING: Skipping HDF5 file {hdf5_path}: {type(e).__name__}: {e!s}")
            continue


class PlanningDataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for planning dataset from HDF5 files."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    N_WORKERS = 10  # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 50  # number of paths converted & stored in memory before writing to disk
    PARSE_FCN = _generate_examples  # handle to parse function from file paths to RLDS episodes

    def _detect_image_shape(self) -> tuple[int, int, int]:
        """Detect image shape from the first image in the first HDF5 file.

        Returns:
            Tuple of (height, width, channels) for the image shape.
        """
        # Get the first HDF5 file path
        hdf5_file = Path(os.getenv("HDF5_FILE_PATH"))
        if not hdf5_file.exists():
            raise FileNotFoundError(f"Data file not found: {hdf5_file}")

        has_overview = False
        # Open the file and get the first image
        with h5py.File(hdf5_file, "r") as f:
            if "data" not in f:
                raise ValueError(f"HDF5 file {hdf5_file} missing 'data' group")

            data_group = f["data"]
            demo_names = sorted(data_group.keys())
            if not demo_names:
                raise ValueError(f"No demos found in {hdf5_file}")

            # Get the first demo
            first_demo = data_group[demo_names[0]]
            obs_group = first_demo["obs"]

            # Get shape from base_image
            if "base_image" not in obs_group:
                raise ValueError("No base_image found in first demo")

            if "overview_image" in obs_group:
                has_overview = True

            base_image_shape = obs_group["base_image"].shape
            # Shape is (T, H, W, C), we want (H, W, C)
            return tuple(base_image_shape[1:]), has_overview

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        # Detect image shape from first image in dataset
        image_shape, has_overview = self._detect_image_shape()

        observation = {
            "base_image": tfds.features.Image(
                shape=image_shape,
                dtype=np.uint8,
                encoding_format="jpeg",
                doc="Base camera RGB observation.",
            ),
            "wrist_image": tfds.features.Image(
                shape=image_shape,
                dtype=np.uint8,
                encoding_format="jpeg",
                doc="Wrist camera RGB observation.",
            ),
            "state": tfds.features.Tensor(
                shape=(11,),
                dtype=np.float32,
                doc="Robot state, consists of [base_pose (3), arm_pos (3), arm_quat (4), gripper_pos (1)].",
            ),
        }
        if has_overview:
            observation["overview_image"] = tfds.features.Image(
                shape=image_shape,
                dtype=np.uint8,
                encoding_format="jpeg",
                doc="Overview camera RGB observation.",
            )

        steps = {
            "observation": tfds.features.FeaturesDict(observation),
            "action": tfds.features.Tensor(
                shape=(10,),
                dtype=np.float32,
                doc="Robot action, 10-dimensional action vector.",
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

        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(steps),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(doc="Path to the original HDF5 file."),
                            "demo_name": tfds.features.Text(doc="Name of the demonstration (e.g., demo_0, demo_1)."),
                        }
                    ),
                }
            )
        )

    def _split_paths(self):
        """Define filepaths for data splits."""
        # TODO: Get actual HDF5 file paths from env variables
        hdf5_file = Path(os.getenv("HDF5_FILE_PATH"))

        if not hdf5_file.exists():
            raise FileNotFoundError(
                f"Data directory not found: {hdf5_file}\nPlease update the path in _split_paths() method."
            )

        return {
            "train": [hdf5_file],
        }
