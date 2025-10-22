import json
from pathlib import Path
from typing import Any, Iterator, Tuple

import cv2
import h5py
import numpy as np
import tensorflow_datasets as tfds

from agibot_dataset.conversion_utils import MultiThreadedDatasetBuilder


def quaternion_to_euler(quat: np.ndarray) -> np.ndarray:
    """
    Convert batched quaternions (x, y, z, w) to Euler angles (rx, ry, rz) in radians.
    - Input: quat with shape (..., 4)
    - Output: euler angles with shape (..., 3)
    - Convention: XYZ intrinsic rotation (roll, pitch, yaw)
    """
    quat = np.asarray(quat)
    assert quat.shape[-1] == 4, (
        "Input must have shape (..., 4) with (x, y, z, w) order."
    )

    qx, qy, qz, qw = np.moveaxis(quat, -1, 0)

    # Normalize quaternion to avoid drift
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    rx = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    ry = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    rz = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack([rx, ry, rz], axis=-1)


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    # the line below needs to be *inside* generate_examples so that each worker creates it's own model
    # creating one shared model outside this function would cause a deadlock
    # _embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _parse_example(episode_path):
        """Parse a single episode.

        episode_path format: /path/to/sample_dataset/observations/{task_id}/{episode_id}
        """
        episode_path = Path(episode_path)
        task_id = episode_path.parent.name
        episode_id = episode_path.name

        # Get base path to sample_dataset
        base_path = episode_path.parent.parent.parent

        # Load task info for language instruction
        task_json = base_path / f"task_info/task_{task_id}.json"
        with open(task_json, "r") as f:
            task_info = json.load(f)[0]
        task_name = task_info["task_name"]
        init_scene = task_info["init_scene_text"]
        language_instruction = f"{task_name}. {init_scene}"

        # Load proprio stats from H5 file
        proprio_path = (
            base_path / f"proprio_stats/{task_id}/{episode_id}/proprio_stats.h5"
        )
        with h5py.File(proprio_path, "r") as f:
            # state_joint = np.array(f["state/joint/position"])
            gripper_state = np.array(f["state/effector/position"])  # shape (N, 2)
            ee_xyz = np.array(f["state/end/position"])  # shape (N, 2, 3)
            ee_xyzw = np.array(f["state/end/orientation"])  # shape (N, 2, 4)
            left_state = np.hstack(
                [
                    ee_xyz[:, 0, :],
                    quaternion_to_euler(ee_xyzw[:, 0, :]),
                    gripper_state[:, 0:1],
                ]
            )  # shape (N, 8)
            right_state = np.hstack(
                [
                    ee_xyz[:, 1, :],
                    quaternion_to_euler(ee_xyzw[:, 1, :]),
                    gripper_state[:, 1:2],
                ]
            )  # shape (N, 8)
            # state_head = np.array(f["state/head/position"])
            # state_waist = np.array(f["state/waist/position"])
            # action_joint = np.array(f["action/joint/position"])
            gripper_action = np.array(f["action/effector/position"])  # shape (N, 2)
            ee_action_xyz = np.array(f["action/end/position"])  # shape (N, 2, 3)
            ee_action_xyzw = np.array(f["action/end/orientation"])  # shape (N, 2, 4)
            left_action = np.hstack(
                [
                    ee_action_xyz[:, 0, :],
                    quaternion_to_euler(ee_action_xyzw[:, 0, :]),
                    gripper_action[:, 0:1],
                ]
            )  # shape (N, 8)
            right_action = np.hstack(
                [
                    ee_action_xyz[:, 1, :],
                    quaternion_to_euler(ee_action_xyzw[:, 1, :]),
                    gripper_action[:, 1:2],
                ]
            )  # shape (N, 8)
            # action_head = np.array(f["action/head/position"])
            # action_waist = np.array(f["action/waist/position"])
            # action_velocity = np.array(f["action/robot/velocity"])

        # Combine states and actions
        states = np.hstack([left_state, right_state]).astype(np.float32)
        actions = np.hstack([left_action, right_action]).astype(np.float32)

        # Load depth images
        depth_dir = episode_path / "depth"
        depth_files = sorted(list(depth_dir.glob("head_depth_*.png")))

        # Verify we have the right number of frames
        num_frames = len(depth_files)
        assert num_frames == len(states), (
            f"Mismatch: {num_frames} depth images vs {len(states)} states"
        )
        assert num_frames == len(actions), (
            f"Mismatch: {num_frames} depth images vs {len(actions)} actions"
        )

        # Compute language embedding once (it's the same for all steps)
        # language_embedding = _embed([language_instruction])[0].numpy()
        # language_embedding = np.zeros((512,), dtype=np.float32)  # dummy embedding

        # Load and decode video frames
        video_dir = episode_path / "videos"
        video_files = {
            "head_color": video_dir / "head_color.mp4",
            "hand_left_color": video_dir / "hand_left_color.mp4",
            "hand_right_color": video_dir / "hand_right_color.mp4",
            "head_center_fisheye_color": video_dir / "head_center_fisheye_color.mp4",
            # 'head_left_fisheye_color': video_dir / 'head_left_fisheye_color.mp4',
            # 'head_right_fisheye_color': video_dir / 'head_right_fisheye_color.mp4',
            # 'back_left_fisheye_color': video_dir / 'back_left_fisheye_color.mp4',
            # 'back_right_fisheye_color': video_dir / 'back_right_fisheye_color.mp4',
        }

        # Read video frames using OpenCV (more reliable with multiprocessing)
        def read_video_frames(video_path):
            """Read all frames from a video file using OpenCV."""
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise IOError(f"Cannot open video file: {video_path}")

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB (OpenCV uses BGR by default)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize to 224x224
                frame_resized = cv2.resize(
                    frame_rgb, (224, 224), interpolation=cv2.INTER_LINEAR
                )
                frames.append(frame_resized)

            cap.release()
            return frames

        # Read all videos
        video_frames = {}
        for key, video_path in video_files.items():
            video_frames[key] = read_video_frames(video_path)

        # Verify all videos have the same number of frames
        for key, frames in video_frames.items():
            assert len(frames) == num_frames, (
                f"Video {key} has {len(frames)} frames, expected {num_frames}"
            )

        # Build episode steps
        episode = []
        for i in range(num_frames):
            # # Load depth image
            # depth_img = (
            #     np.array(Image.open(depth_files[i])).astype(np.float32) / 1000.0
            # )  # Convert to meters
            # # Add channel dimension
            # depth_img = np.expand_dims(depth_img, axis=-1)

            episode.append(
                {
                    "observation": {
                        "head_image": video_frames["head_color"][i],
                        "hand_left_image": video_frames["hand_left_color"][i],
                        "hand_right_image": video_frames["hand_right_color"][i],
                        "head_center_fisheye_image": video_frames[
                            "head_center_fisheye_color"
                        ][i],
                        # "head_left_fisheye_image": video_frames[
                        #     "head_left_fisheye_color"
                        # ][i],
                        # "head_right_fisheye_image": video_frames[
                        #     "head_right_fisheye_color"
                        # ][i],
                        # "back_left_fisheye_image": video_frames[
                        #     "back_left_fisheye_color"
                        # ][i],
                        # "back_right_fisheye_image": video_frames[
                        #     "back_right_fisheye_color"
                        # ][i],
                        # "depth_image": depth_img,
                        "state": states[i],
                    },
                    "action": actions[i],
                    "discount": 1.0,
                    "reward": float(i == (num_frames - 1)),
                    "is_first": i == 0,
                    "is_last": i == (num_frames - 1),
                    "is_terminal": i == (num_frames - 1),
                    "language_instruction": language_instruction,
                    # "language_embedding": language_embedding,
                }
            )

        # Create output data sample
        sample = {
            "steps": episode,
            "episode_metadata": {
                "file_path": str(episode_path),
                "task_id": int(task_id),
                "episode_id": int(episode_id),
            },
        }

        # Return with a unique key
        return f"{task_id}_{episode_id}", sample

    # Parse examples from paths
    for sample in paths:
        yield _parse_example(sample)


class AgibotDataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for sample humanoid robot dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    N_WORKERS = 1  # number of parallel workers for data conversion
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
                                    # "head_left_fisheye_image": tfds.features.Image(
                                    #     shape=(748, 960, 3),
                                    #     dtype=np.uint8,
                                    #     encoding_format="png",
                                    #     doc="Head left fisheye camera RGB observation.",
                                    # ),
                                    # "head_right_fisheye_image": tfds.features.Image(
                                    #     shape=(748, 960, 3),
                                    #     dtype=np.uint8,
                                    #     encoding_format="png",
                                    #     doc="Head right fisheye camera RGB observation.",
                                    # ),
                                    # "back_left_fisheye_image": tfds.features.Image(
                                    #     shape=(748, 960, 3),
                                    #     dtype=np.uint8,
                                    #     encoding_format="png",
                                    #     doc="Back left fisheye camera RGB observation.",
                                    # ),
                                    # "back_right_fisheye_image": tfds.features.Image(
                                    #     shape=(748, 960, 3),
                                    #     dtype=np.uint8,
                                    #     encoding_format="png",
                                    #     doc="Back right fisheye camera RGB observation.",
                                    # ),
                                    # "depth_image": tfds.features.Tensor(
                                    #     shape=(480, 640, 1),
                                    #     dtype=np.float32,
                                    #     doc="Head depth camera observation in meters.",
                                    # ),
                                    "state": tfds.features.Tensor(
                                        shape=(14,),
                                        dtype=np.float32,
                                        doc="Robot state, consists of [left ee pose, left gripper, right ee pose, right gripper]. ee pose is [x, y, z, r, p, y]. Gripper open range in mm. xyz in meters.",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(14,),
                                dtype=np.float32,
                                doc="Robot action, consists of [left ee action, left gripper action, right ee action, right gripper action]. ee action is [x, y, z, r, p, y]. Gripper 0 for full open and 1 for full close. xyz in meters.",
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
                            # "language_embedding": tfds.features.Tensor(
                            #     shape=(512,),
                            #     dtype=np.float32,
                            #     doc="Kona language embedding. "
                            #     "See https://tfhub.dev/google/universal-sentence-encoder-large/5",
                            # ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(
                                doc="Path to the original data file."
                            ),
                            "task_id": tfds.features.Scalar(
                                dtype=np.int32, doc="Task ID."
                            ),
                            "episode_id": tfds.features.Scalar(
                                dtype=np.int32, doc="Episode ID."
                            ),
                        }
                    ),
                }
            )
        )

    def _split_paths(self):
        """Define filepaths for data splits."""
        # Find all episode directories in sample_dataset/observations
        base_path = Path("/n/fs/robot-data/rlds_multithread/agibot_dataset/data")
        all_episodes = []

        # Iterate through all task directories
        for task_dir in sorted((base_path / "observations").iterdir()):
            if task_dir.is_dir():
                # Iterate through all episode directories
                for episode_dir in sorted(task_dir.iterdir()):
                    if episode_dir.is_dir():
                        all_episodes.append(str(episode_dir))

        print(f"Found {len(all_episodes)} episodes")
        return {
            "train": all_episodes,
        }
