import io
import json
from pathlib import Path
from typing import Any, Iterator, Tuple

import numpy as np
import pyarrow.parquet as pq
import tensorflow_datasets as tfds
from PIL import Image

from agibot_dataset.conversion_utils import MultiThreadedDatasetBuilder


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""

    def _parse_example(episode_path):
        """Parse a single episode from a parquet file.

        episode_path format: /path/to/data/chunk-XXX/episode_XXXXXX.parquet
        """
        episode_path = Path(episode_path)

        # Load the parquet file
        table = pq.read_table(str(episode_path))

        # Extract metadata
        episode_index = int(table["episode_index"][0].as_py())
        task_index = int(table["task_index"][0].as_py())

        # Get the base directory (train or val) and dataset name
        base_path = episode_path.parent.parent.parent
        # Extract dataset name (molmoact_dataset_household or molmoact_dataset_tabletop)
        dataset_name = base_path.parent.name

        # Load task information
        tasks_path = base_path / "meta" / "tasks.jsonl"
        task_name = None
        with open(tasks_path, "r") as f:
            for line in f:
                task_info = json.loads(line)
                if task_info["task_index"] == task_index:
                    task_name = task_info["task"]
                    break

        if task_name is None:
            task_name = f"task_{task_index}"

        language_instruction = task_name

        # Convert table to lists
        num_frames = len(table)

        # Extract images
        first_view_data = table["first_view"].to_pylist()
        second_view_data = table["second_view"].to_pylist()
        wrist_image_data = table["wrist_image"].to_pylist()

        # Extract states and actions
        states = np.array(table["state"].to_pylist(), dtype=np.float32)  # shape (N, 7)
        actions = np.array(
            table["actions"].to_pylist(), dtype=np.float32
        )  # shape (N, 7)

        # Decode images and resize to 224x224
        def decode_and_resize_image(img_struct):
            """Decode image from bytes and resize to 224x224."""
            img_bytes = img_struct["bytes"]
            img = Image.open(io.BytesIO(img_bytes))
            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")
            # Resize to 224x224
            img_resized = img.resize((224, 224), Image.BILINEAR)
            return np.array(img_resized, dtype=np.uint8)

        # Decode all images
        first_view_images = [decode_and_resize_image(img) for img in first_view_data]
        second_view_images = [decode_and_resize_image(img) for img in second_view_data]
        wrist_images = [decode_and_resize_image(img) for img in wrist_image_data]

        # Build episode steps
        episode = []
        for i in range(num_frames):
            episode.append(
                {
                    "observation": {
                        "first_view_image": first_view_images[i],
                        "second_view_image": second_view_images[i],
                        "wrist_image": wrist_images[i],
                        "state": states[i],
                    },
                    "action": actions[i],
                    "discount": 1.0,
                    "reward": float(i == (num_frames - 1)),
                    "is_first": i == 0,
                    "is_last": i == (num_frames - 1),
                    "is_terminal": i == (num_frames - 1),
                    "language_instruction": language_instruction,
                }
            )

        # Create output data sample
        sample = {
            "steps": episode,
            "episode_metadata": {
                "file_path": str(episode_path),
                "episode_index": episode_index,
                "task_index": task_index,
            },
        }

        # Return with a unique key that includes dataset name to avoid collisions
        # when combining multiple datasets
        return f"{dataset_name}_{episode_index}", sample

    # Parse examples from paths
    for sample in paths:
        yield _parse_example(sample)


class MolmoactDataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for MolmoAct robot dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    N_WORKERS = 30  # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = (
        100  # number of paths converted & stored in memory before writing to disk
    )
    PARSE_FCN = (
        _generate_examples  # handle to parse function from file paths to RLDS episodes
    )

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "first_view_image": tfds.features.Image(
                                        shape=(224, 224, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="First view camera RGB observation.",
                                    ),
                                    "second_view_image": tfds.features.Image(
                                        shape=(224, 224, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Second view camera RGB observation.",
                                    ),
                                    "wrist_image": tfds.features.Image(
                                        shape=(224, 224, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Wrist camera RGB observation.",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc="Robot state, consists of [x, y, z, rx, ry, rz, gripper]. Positions in meters, rotations in radians, gripper in mm.",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(7,),
                                dtype=np.float32,
                                doc="Robot action, consists of [dx, dy, dz, drx, dry, drz, gripper]. Deltas in meters and radians.",
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
                            "episode_index": tfds.features.Scalar(
                                dtype=np.int32, doc="Episode index."
                            ),
                            "task_index": tfds.features.Scalar(
                                dtype=np.int32, doc="Task index."
                            ),
                        }
                    ),
                }
            )
        )

    def _split_paths(self):
        """Define filepaths for data splits."""
        # Find all parquet files from both household and tabletop datasets
        base_dir = Path("/n/fs/robot-data/data/molmoact-dataset/data")

        dataset_names = ["molmoact_dataset_household", "molmoact_dataset_tabletop"]
        all_episodes = []

        for dataset_name in dataset_names:
            dataset_path = base_dir / dataset_name / "train" / "data"
            if dataset_path.exists():
                # Find all parquet files in all chunks
                parquet_files = sorted(dataset_path.glob("**/*.parquet"))
                all_episodes.extend([str(f) for f in parquet_files])
                print(f"Found {len(parquet_files)} episodes in {dataset_name}")

        print(f"Total: {len(all_episodes)} episodes across all datasets")
        return {
            "train": all_episodes,
        }
