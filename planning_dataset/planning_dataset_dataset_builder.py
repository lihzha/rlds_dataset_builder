import pickle
from pathlib import Path
from typing import Any, Iterator, Tuple

import cv2
import numpy as np
import tensorflow_datasets as tfds

from planning_dataset.conversion_utils import MultiThreadedDatasetBuilder


def read_video_frames(video_path: Path) -> list[np.ndarray]:
    """Read all frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""

    def _parse_example(demo_path):
        """Parse a single demonstration episode.

        demo_path format: /path/to/demos/{demo_name}
        """
        demo_path = Path(demo_path)
        demo_name = demo_path.name

        # Load required files
        pkl_path = demo_path / "data.pkl"
        base_video_path = demo_path / "base_image.mp4"
        wrist_video_path = demo_path / "wrist_image.mp4"

        # Check if all required files exist
        if not pkl_path.exists():
            raise FileNotFoundError(f"data.pkl not found in {demo_path}")
        if not base_video_path.exists():
            raise FileNotFoundError(f"base_image.mp4 not found in {demo_path}")
        if not wrist_video_path.exists():
            raise FileNotFoundError(f"wrist_image.mp4 not found in {demo_path}")

        # Load pickle data
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        timestamps = data["timestamps"]
        observations = data["observations"]
        actions = data["actions"]

        # Read video frames
        base_frames = read_video_frames(base_video_path)
        wrist_frames = read_video_frames(wrist_video_path)

        # Verify alignment
        num_timesteps = len(timestamps)
        num_base_frames = len(base_frames)
        num_wrist_frames = len(wrist_frames)

        if not (num_timesteps == num_base_frames == num_wrist_frames):
            raise ValueError(
                f"Misalignment in {demo_name}: "
                f"timesteps={num_timesteps}, base_frames={num_base_frames}, "
                f"wrist_frames={num_wrist_frames}"
            )

        # Build episode steps
        episode = []
        for i in range(num_timesteps):
            obs = observations[i]
            action = actions[i]

            # Construct state: arm_pos (3) + arm_quat (4) + gripper_pos (1) = 8
            state = np.concatenate(
                [
                    obs["arm_pos"],  # (3,)
                    obs["arm_quat"],  # (4,)
                    obs["gripper_pos"],  # (1,)
                ]
            ).astype(np.float32)

            # Construct action: arm_pos (3) + arm_quat (4) + gripper_pos (1) = 8
            action_vector = np.concatenate(
                [
                    action["arm_pos"],  # (3,)
                    action["arm_quat"],  # (4,)
                    action["gripper_pos"],  # (1,)
                ]
            ).astype(np.float32)

            # Add step to episode
            episode.append(
                {
                    "observation": {
                        "base_image": base_frames[i],
                        "wrist_image": wrist_frames[i],
                        "state": state,
                    },
                    "action": action_vector,
                    "discount": 1.0,
                    "reward": float(i == (num_timesteps - 1)),
                    "is_first": i == 0,
                    "is_last": i == (num_timesteps - 1),
                    "is_terminal": i == (num_timesteps - 1),
                    "language_instruction": f"tiger_demo_{demo_name}",
                }
            )

        # Create output data sample
        sample = {
            "steps": episode,
            "episode_metadata": {
                "file_path": str(demo_path),
                "demo_name": demo_name,
            },
        }

        return demo_name, sample

    # Parse examples from paths
    for sample in paths:
        try:
            unique_key, episode_data = _parse_example(sample)
            yield unique_key, episode_data
        except Exception as e:
            # Extract demo name from path
            demo_path = Path(sample)
            demo_name = demo_path.name

            # Log the error
            print(f"WARNING: Skipping demo {demo_name}: {type(e).__name__}: {str(e)}")
            continue


class PlanningDataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for planning dataset."""

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
                                    "base_image": tfds.features.Image(
                                        shape=(360, 640, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Base camera RGB observation.",
                                    ),
                                    "wrist_image": tfds.features.Image(
                                        shape=(480, 640, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Wrist camera RGB observation.",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(8,),
                                        dtype=np.float32,
                                        doc="Robot state, consists of [arm_pos (3), arm_quat (4), gripper_pos (1)].",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(8,),
                                dtype=np.float32,
                                doc="Robot action, consists of [arm_pos (3), arm_quat (4), gripper_pos (1)].",
                            ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Discount if provided, default to 1.",
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Reward if provided, 1 on final step for demos.",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on first step of the episode."
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on last step of the episode."
                            ),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode if it is a terminal step, True for demos.",
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Language Instruction."
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(
                                doc="Path to the original data file."
                            ),
                            "demo_name": tfds.features.Text(
                                doc="Name of the demonstration."
                            ),
                        }
                    ),
                }
            )
        )

    def _split_paths(self):
        """Define filepaths for data splits."""
        # TODO: Update this path to point to your actual data directory
        base_path = Path("/path/to/your/tiger/demos")

        if not base_path.exists():
            raise FileNotFoundError(
                f"Data directory not found: {base_path}\n"
                "Please update the path in _split_paths() method."
            )

        # Get all demo directories
        all_demos = sorted([str(d) for d in base_path.iterdir() if d.is_dir()])

        print(f"Found {len(all_demos)} demonstrations")

        return {
            "train": all_demos,
        }
