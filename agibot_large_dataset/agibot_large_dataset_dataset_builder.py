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
    - Output: Euler angles with shape (..., 3)
    - Convention: Extrinsic XYZ rotation (i.e. equivalent to intrinsic ZYX)
    """
    quat = np.asarray(quat)
    assert quat.shape[-1] == 4, (
        "Input must have shape (..., 4) with (x, y, z, w) order."
    )

    qx, qy, qz, qw = np.moveaxis(quat, -1, 0)

    # Normalize
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    # Equivalent to intrinsic ZYX (yaw, pitch, roll)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy**2 + qz**2)
    rz = np.arctan2(siny_cosp, cosy_cosp)  # yaw (Z)

    sinp = 2 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    ry = np.arcsin(sinp)  # pitch (Y)

    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx**2 + qy**2)
    rx = np.arctan2(sinr_cosp, cosr_cosp)  # roll (X)

    return np.stack([rx, ry, rz], axis=-1)


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    # the line below needs to be *inside* generate_examples so that each worker creates it's own model
    # creating one shared model outside this function would cause a deadlock
    # _embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    # Track corrupted episodes
    corrupted_episodes_file = Path(
        "/n/fs/robot-data/rlds_multithread/corrupted_episodes.json"
    )

    def _log_corrupted_episode(task_id, episode_id, error_msg):
        """Log corrupted episode to file."""
        if corrupted_episodes_file.exists():
            with open(corrupted_episodes_file, "r") as f:
                corrupted_data = json.load(f)
        else:
            corrupted_data = {"corrupted_episodes": []}

        # Add new corrupted episode
        corrupted_entry = {
            "task_id": task_id,
            "episode_id": episode_id,
            "error": error_msg,
            "timestamp": str(Path(__file__).stat().st_mtime),
        }

        # Check if already logged
        existing = [
            e
            for e in corrupted_data["corrupted_episodes"]
            if e["task_id"] == task_id and e["episode_id"] == episode_id
        ]

        if not existing:
            corrupted_data["corrupted_episodes"].append(corrupted_entry)
            with open(corrupted_episodes_file, "w") as f:
                json.dump(corrupted_data, f, indent=2)

        print(
            f"WARNING: Skipping corrupted episode {task_id}/{episode_id}: {error_msg}"
        )

    def _parse_example(episode_path):
        """Parse a single episode and split into subskill episodes.

        episode_path format: /path/to/AgiBotWorld-Beta-New/observations/{task_id}/{episode_id}
        """
        episode_path = Path(episode_path)
        task_id = episode_path.parent.name
        episode_id = episode_path.name

        # Get base path to AgiBotWorld-Beta
        base_path = episode_path.parent.parent.parent

        # Load task info for this specific episode
        task_json = base_path / f"task_info/task_{task_id}.json"
        with open(task_json, "r") as f:
            task_data = json.load(f)

        # Find the task_info entry for this specific episode_id
        episode_task_info = None
        for entry in task_data:
            if str(entry["episode_id"]) == episode_id:
                episode_task_info = entry
                break

        if episode_task_info is None:
            raise ValueError(f"Episode {episode_id} not found in task {task_id} JSON")

        task_name = episode_task_info["task_name"]
        action_configs = episode_task_info.get("label_info", {}).get(
            "action_config", []
        )

        # Load proprio stats from H5 file
        proprio_path = (
            base_path / f"proprio_stats/{task_id}/{episode_id}/proprio_stats.h5"
        )
        with h5py.File(proprio_path, "r") as f:
            # state_joint = np.array(f["state/joint/position"])
            gripper_state = np.array(f["state/effector/position"])  # shape (N, 2)
            ee_xyz = np.array(f["state/end/position"])  # shape (N, 2, 3)
            ee_xyzw = np.array(f["state/end/orientation"])  # shape (N, 2, 4)

            # Validate shape of proprioception data
            expected_shapes = {
                "gripper_state": (None, 2),
                "ee_xyz": (None, 2, 3),
                "ee_xyzw": (None, 2, 4),
            }

            for name, arr, expected in [
                ("gripper_state", gripper_state, expected_shapes["gripper_state"]),
                ("ee_xyz", ee_xyz, expected_shapes["ee_xyz"]),
                ("ee_xyzw", ee_xyzw, expected_shapes["ee_xyzw"]),
            ]:
                actual_shape = arr.shape
                expected_dims = len(expected)
                if len(actual_shape) != expected_dims:
                    raise ValueError(
                        f"Proprioception data shape error: {name} has shape {actual_shape}, "
                        f"expected {expected_dims} dimensions"
                    )
                # Check non-batch dimensions
                for i in range(1, expected_dims):
                    if actual_shape[i] != expected[i]:
                        raise ValueError(
                            f"Proprioception data shape error: {name} has shape {actual_shape}, "
                            f"expected dimension {i} to be {expected[i]}"
                        )

            # Validate proprioception data consistency
            state_lengths = {
                "gripper_state": len(gripper_state),
                "ee_xyz": len(ee_xyz),
                "ee_xyzw": len(ee_xyzw),
            }

            gripper_action = np.array(f["action/effector/position"])  # shape (N, 2)
            ee_action_xyz = np.array(f["action/end/position"])  # shape (N, 2, 3)
            ee_action_xyzw = np.array(f["action/end/orientation"])  # shape (N, 2, 4)

            # Validate action data shapes
            for name, arr, expected in [
                ("gripper_action", gripper_action, expected_shapes["gripper_state"]),
                ("ee_action_xyz", ee_action_xyz, expected_shapes["ee_xyz"]),
                ("ee_action_xyzw", ee_action_xyzw, expected_shapes["ee_xyzw"]),
            ]:
                actual_shape = arr.shape
                expected_dims = len(expected)
                if len(actual_shape) != expected_dims:
                    raise ValueError(
                        f"Proprioception data shape error: {name} has shape {actual_shape}, "
                        f"expected {expected_dims} dimensions"
                    )
                # Check non-batch dimensions
                for i in range(1, expected_dims):
                    if actual_shape[i] != expected[i]:
                        raise ValueError(
                            f"Proprioception data shape error: {name} has shape {actual_shape}, "
                            f"expected dimension {i} to be {expected[i]}"
                        )

            action_lengths = {
                "gripper_action": len(gripper_action),
                "ee_action_xyz": len(ee_action_xyz),
                "ee_action_xyzw": len(ee_action_xyzw),
            }

            # Check if all proprio data has same length
            all_lengths = list(state_lengths.values()) + list(action_lengths.values())
            if len(set(all_lengths)) > 1:
                state_details = ", ".join(
                    [f"{k}={v}" for k, v in state_lengths.items()]
                )
                action_details = ", ".join(
                    [f"{k}={v}" for k, v in action_lengths.items()]
                )
                raise ValueError(
                    f"Proprioception data length mismatch: "
                    f"state: {state_details}; action: {action_details}"
                )

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

        num_states = len(states)
        num_actions = len(actions)

        # # Load depth images
        # depth_dir = episode_path / "depth"
        # depth_files = sorted(list(depth_dir.glob("head_depth_*.png")))

        # # Verify we have the right number of frames
        # num_frames = len(depth_files)
        # assert num_frames == len(states), (
        #     f"Mismatch: {num_frames} depth images vs {len(states)} states"
        # )
        # assert num_frames == len(actions), (
        #     f"Mismatch: {num_frames} depth images vs {len(actions)} actions"
        # )

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
            if not video_path.exists():
                raise IOError(f"Video file not found: {video_path}")

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise IOError(
                    f"Cannot open video file (possibly corrupted): {video_path}"
                )

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

        # Check frame count consistency
        video_frame_counts = {key: len(frames) for key, frames in video_frames.items()}
        all_frame_counts = list(video_frame_counts.values())

        # Find the minimum frame count (most restrictive)
        min_frame_count = min(all_frame_counts)
        max_frame_count = max(all_frame_counts)

        # If frame counts don't match, raise an error with details
        if min_frame_count != max_frame_count:
            frame_count_details = ", ".join(
                [f"{k}={v}" for k, v in video_frame_counts.items()]
            )
            raise ValueError(
                f"Frame count mismatch: videos: {frame_count_details}. "
                f"Min={min_frame_count}, Max={max_frame_count}"
            )

        assert min_frame_count == num_actions
        assert min_frame_count == num_states
        assert num_actions == num_states

        # Process each subskill as a separate episode
        subskill_episodes = []

        for action_config in action_configs:
            start_frame = action_config["start_frame"]
            end_frame = action_config["end_frame"]
            action_text = action_config.get("action_text", "").strip()
            skill_name = action_config.get("skill", "Unknown")

            # Generate language instruction
            if action_text:
                language_instruction = action_text
            else:
                language_instruction = f"Episode {episode_id}_frame_{start_frame}"

            # Build episode steps for this subskill
            episode = []
            subskill_num_frames = end_frame - start_frame

            for i in range(start_frame, end_frame):
                step_idx = i - start_frame  # 0-indexed step within subskill

                episode.append(
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
                        "reward": float(step_idx == (subskill_num_frames - 1)),
                        "is_first": step_idx == 0,
                        "is_last": step_idx == (subskill_num_frames - 1),
                        "is_terminal": step_idx == (subskill_num_frames - 1),
                        "language_instruction": language_instruction,
                    }
                )

            # Create output data sample for this subskill
            sample = {
                "steps": episode,
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

            # Create unique key with start_frame
            unique_key = f"{task_id}_{episode_id}_{start_frame}"
            subskill_episodes.append((unique_key, sample))

        # Return all subskill episodes
        return subskill_episodes

    # Parse examples from paths
    for sample in paths:
        try:
            subskill_episodes = _parse_example(sample)
            # Yield each subskill episode separately
            for unique_key, episode_data in subskill_episodes:
                yield unique_key, episode_data
        except Exception as e:
            # Extract task_id and episode_id from path
            episode_path = Path(sample)
            task_id = episode_path.parent.name
            episode_id = episode_path.name

            # Log the corrupted episode
            error_msg = f"{type(e).__name__}: {str(e)}"
            # _log_corrupted_episode(task_id, episode_id, error_msg)

            # Continue processing other episodes
            continue


class AgibotLargeDataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for sample humanoid robot dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    N_WORKERS = 40  # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = (
        50  # number of paths converted & stored in memory before writing to disk
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
                                dtype=np.int32,
                                doc="Unique episode ID (task_id_original_episode_id_start_frame).",
                            ),
                            "original_episode_id": tfds.features.Scalar(
                                dtype=np.int32,
                                doc="Original episode ID before subskill splitting.",
                            ),
                            "start_frame": tfds.features.Scalar(
                                dtype=np.int32,
                                doc="Start frame of this subskill in the original episode.",
                            ),
                            "end_frame": tfds.features.Scalar(
                                dtype=np.int32,
                                doc="End frame of this subskill in the original episode.",
                            ),
                            "subskill_name": tfds.features.Text(
                                doc="Name of the subskill (e.g., Pick, Place, Pour)."
                            ),
                        }
                    ),
                }
            )
        )

    def _split_paths(self):
        """Define filepaths for data splits."""
        # Load tracking file to get processable episodes
        tracking_file = Path(
            "/n/fs/robot-data/rlds_multithread/processed_episodes.json"
        )

        if not tracking_file.exists():
            raise FileNotFoundError(
                f"Tracking file not found: {tracking_file}\n"
                "Please run scan_and_track_episodes.py first to generate the tracking file."
            )

        with open(tracking_file, "r") as f:
            tracking_data = json.load(f)

        base_path = Path("/n/fs/robot-data/data/AgiBotWorld-Beta-New")
        all_episodes = []

        processed_episodes = tracking_data.get("processed_episodes", {})

        # Limit to first 10 episodes per task
        MAX_EPISODES_PER_TASK = 10

        # Iterate through all processable tasks
        for task_id in sorted(processed_episodes.keys()):
            task_dir = base_path / "observations" / task_id

            if not task_dir.is_dir():
                print(f"Warning: Task directory not found: {task_dir}")
                continue

            task_episode_count = 0

            # Iterate through episode IDs for this task (limited to first 50)
            for episode_id in sorted(processed_episodes[task_id]):
                if task_episode_count >= MAX_EPISODES_PER_TASK:
                    break

                episode_dir = task_dir / episode_id

                # Verify the episode directory exists and has required structure
                if episode_dir.is_dir():
                    videos_dir = episode_dir / "videos"
                    # depth_dir = episode_dir / "depth"

                    if videos_dir.exists():
                        all_episodes.append(str(episode_dir))
                        task_episode_count += 1
                    else:
                        print(
                            f"Warning: Episode {episode_dir} missing videos or depth directory"
                        )

        print(f"Found {len(all_episodes)} episodes to process (max 10 per task)")
        print(
            f"Skipped {len(tracking_data.get('skipped_tasks', []))} tasks (only tar files)"
        )

        return {
            "train": all_episodes,
        }
