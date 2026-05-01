"""Canonical FeaturesDict for the AgiBot large dataset.

Used by:
- process_task.py to encode/serialize examples into tfrecord shards
- merge_shards.py to write features.json into the final dataset dir

Keep this in sync with agibot_large_dataset/agibot_large_dataset_dataset_builder.py.
"""

import numpy as np
import tensorflow_datasets as tfds


DATASET_NAME = "agibot_large_dataset"
DATASET_VERSION = "1.0.0"
SPLIT_NAME = "train"


def build_features() -> tfds.features.FeaturesDict:
    return tfds.features.FeaturesDict(
        {
            "steps": tfds.features.Dataset(
                {
                    "observation": tfds.features.FeaturesDict(
                        {
                            "head_image": tfds.features.Image(
                                shape=(224, 224, 3),
                                dtype=np.uint8,
                                encoding_format="jpeg",
                                doc="Head camera RGB observation.",
                            ),
                            "hand_left_image": tfds.features.Image(
                                shape=(224, 224, 3),
                                dtype=np.uint8,
                                encoding_format="jpeg",
                                doc="Left hand camera RGB observation.",
                            ),
                            "hand_right_image": tfds.features.Image(
                                shape=(224, 224, 3),
                                dtype=np.uint8,
                                encoding_format="jpeg",
                                doc="Right hand camera RGB observation.",
                            ),
                            "head_center_fisheye_image": tfds.features.Image(
                                shape=(224, 224, 3),
                                dtype=np.uint8,
                                encoding_format="jpeg",
                                doc="Head center fisheye camera RGB observation.",
                            ),
                            "state": tfds.features.Tensor(
                                shape=(14,),
                                dtype=np.float32,
                                doc=(
                                    "Robot state: [left ee pose (x,y,z,r,p,y), left gripper, "
                                    "right ee pose (x,y,z,r,p,y), right gripper]. "
                                    "xyz in meters, gripper open range in mm."
                                ),
                            ),
                        }
                    ),
                    "action": tfds.features.Tensor(
                        shape=(14,),
                        dtype=np.float32,
                        doc=(
                            "Robot action: [left ee action (x,y,z,r,p,y), left gripper action, "
                            "right ee action (x,y,z,r,p,y), right gripper action]. "
                            "xyz in meters, gripper 0=open, 1=close."
                        ),
                    ),
                    "discount": tfds.features.Scalar(
                        dtype=np.float32, doc="Discount, default 1.0."
                    ),
                    "reward": tfds.features.Scalar(
                        dtype=np.float32, doc="Reward, 1.0 on final step."
                    ),
                    "is_first": tfds.features.Scalar(
                        dtype=np.bool_, doc="True on first step of the episode."
                    ),
                    "is_last": tfds.features.Scalar(
                        dtype=np.bool_, doc="True on last step of the episode."
                    ),
                    "is_terminal": tfds.features.Scalar(
                        dtype=np.bool_, doc="True on last step (terminal for demos)."
                    ),
                    "language_instruction": tfds.features.Text(
                        doc="Language instruction for the subskill."
                    ),
                }
            ),
            "episode_metadata": tfds.features.FeaturesDict(
                {
                    "file_path": tfds.features.Text(doc="Original episode path."),
                    "task_id": tfds.features.Scalar(dtype=np.int32, doc="Task ID."),
                    "episode_id": tfds.features.Scalar(
                        dtype=np.int32, doc="Episode ID."
                    ),
                    "original_episode_id": tfds.features.Scalar(
                        dtype=np.int32, doc="Original episode ID."
                    ),
                    "start_frame": tfds.features.Scalar(
                        dtype=np.int32, doc="Subskill start frame."
                    ),
                    "end_frame": tfds.features.Scalar(
                        dtype=np.int32, doc="Subskill end frame."
                    ),
                    "subskill_name": tfds.features.Text(
                        doc="Subskill name (Pick, Place, ...)."
                    ),
                }
            ),
        }
    )
