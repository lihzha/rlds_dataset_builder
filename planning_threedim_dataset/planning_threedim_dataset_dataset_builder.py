from collections.abc import Iterator
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import tensorflow_datasets as tfds

from planning_threedim_dataset.conversion_utils import MultiThreadedDatasetBuilder

_LANGUAGE_INSTRUCTIONS = {
    "basemotion3d": "Reach the goal",
    "stickbutton2d": "Use the stick to touch all buttons",
    "dynobstruction2d": "Place a target block onto a target surface",
    "dynpushpullhook2d_o5": "Use a hook to move a target block onto a middle wall",
    "transport3d": "Place all the objects on the table using the box",
    "shelf3d": "Pick the object and place it on the shelf",
    "sweep": "Open the drawer and sweep all the objects into the drawer",
    "motion2d_p0": "Reach the goal",
}


def _get_language_instruction(hdf5_path: Path) -> str:
    stem = hdf5_path.stem.lower()
    match = max(
        (key for key in _LANGUAGE_INSTRUCTIONS if stem.startswith(key)),
        key=len,
        default=None,
    )
    if match is None:
        raise ValueError(f"No language instruction found for {hdf5_path.name}")
    return _LANGUAGE_INSTRUCTIONS[match]


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
                robot_state = np.array(obs_group["robot_state"])  # shape (T, 19)
                base_image = np.array(obs_group["base_image"])  # shape (T, 84, 84, 3)
                wrist_image = np.array(obs_group["wrist_image"])  # shape (T, 84, 84, 3)
                if "overview_image" in obs_group:
                    overview_image = np.array(obs_group["overview_image"])  # shape (T, 84, 84, 3)

                language_instruction = _get_language_instruction(hdf5_path)

                # Verify data alignment
                num_timesteps = len(actions)
                if not (len(base_image) == len(wrist_image) == num_timesteps):
                    raise ValueError(
                        f"Data misalignment in {hdf5_path}/{demo_name}: "
                        f"robot_state={len(robot_state)}, "
                        f"base_image={len(base_image)}, wrist_image={len(wrist_image)}"
                    )

                # Build episode steps
                episode = []
                for i in range(num_timesteps):
                    # Actions are 10-dimensional, we'll store them as-is
                    action_vector = actions[i].astype(np.float32)
                    observation_dict = {
                        "base_image": base_image[i].astype(np.uint8),
                        "wrist_image": wrist_image[i].astype(np.uint8),
                        "state": robot_state[i].astype(np.float32),
                    }
                    if "overview_image" in obs_group:
                        observation_dict["overview_image"] = overview_image[i].astype(np.uint8)
                    # Add step to episode
                    episode.append(
                        {
                            "observation": observation_dict,
                            "action": action_vector,
                            "discount": 1.0,
                            "reward": float(i == (num_timesteps - 1)),
                            "is_first": i == 0,
                            "is_last": i == (num_timesteps - 1),
                            "is_terminal": i == (num_timesteps - 1),
                            "language_instruction": language_instruction,
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


class PlanningThreedimDataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for planning dataset from HDF5 files."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    N_WORKERS = 10  # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 50  # number of paths converted & stored in memory before writing to disk
    PARSE_FCN = _generate_examples  # handle to parse function from file paths to RLDS episodes

    def _detect_shapes(self) -> dict:
        """Detect image and robot_state shapes from the first HDF5 file.

        Returns:
            Dict with 'image_shape', 'robot_state_dim', and 'has_overview'.
        """
        # Get the first HDF5 file path
        hdf5_file = Path(os.getenv("HDF5_FILE_PATH"))
        if not hdf5_file.exists():
            raise FileNotFoundError(f"Data file not found: {hdf5_file}")

        has_overview = False
        # Open the file and get the shapes
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
            image_shape = tuple(base_image_shape[1:])

            # Detect robot_state dimension
            if "robot_state" not in obs_group:
                raise ValueError("No robot_state found in first demo")

            robot_state_shape = obs_group["robot_state"].shape
            # Shape is (T, D), we want D
            robot_state_dim = robot_state_shape[1]

            actions_shape = first_demo["actions"].shape
            action_dim = actions_shape[1]

            return {
                "image_shape": image_shape,
                "robot_state_dim": robot_state_dim,
                "action_dim": action_dim,
                "has_overview": has_overview,
            }

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        # Detect shapes from first demo in dataset
        shapes = self._detect_shapes()
        image_shape = shapes["image_shape"]
        robot_state_dim = shapes["robot_state_dim"]
        action_dim = shapes["action_dim"]
        has_overview = shapes["has_overview"]

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
                shape=(robot_state_dim,),
                dtype=np.float32,
                doc=f"Robot state, {robot_state_dim}-dimensional state vector.",
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
                shape=(action_dim,),
                dtype=np.float32,
                doc=f"Robot action, {action_dim}-dimensional action vector.",
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
