from collections.abc import Iterator
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
import tensorflow_datasets as tfds

from franka_dataset.conversion_utils import MultiThreadedDatasetBuilder

# @tf.function
# def _R_from_euler_xyz(angles):
#     """Extrinsic XYZ: R = Rx(roll) @ Ry(pitch) @ Rz(yaw)."""
#     angles = tf.convert_to_tensor(angles)
#     # Ensure last dim is 3
#     roll = angles[..., 0]
#     pitch = angles[..., 1]
#     yaw = angles[..., 2]
#     return tf.linalg.matmul(tf.linalg.matmul(_rot_x(roll), _rot_y(pitch)), _rot_z(yaw))


# @tf.function
# def _euler_xyz_from_R(R, eps=1e-6):
#     """
#     Extract extrinsic XYZ (roll, pitch, yaw) from rotation matrix R.
#     Handles gimbal lock via elementwise tf.where (graph-safe).

#     For extrinsic XYZ: R = Rx(roll) @ Ry(pitch) @ Rz(yaw)
#     Matrix elements: r02 = sin(pitch), r12 = -sin(roll)cos(pitch), r22 = cos(roll)cos(pitch)
#                      r01 = -cos(pitch)sin(yaw), r00 = cos(pitch)cos(yaw)
#     """
#     R = tf.convert_to_tensor(R)
#     dtype = R.dtype
#     eps_t = tf.cast(eps, dtype)
#     zero = tf.zeros([], dtype)
#     one = tf.ones([], dtype)

#     r00 = R[..., 0, 0]
#     r01 = R[..., 0, 1]
#     r02 = R[..., 0, 2]
#     r10 = R[..., 1, 0]
#     r11 = R[..., 1, 1]
#     r12 = R[..., 1, 2]
#     r22 = R[..., 2, 2]

#     # Regular case: |r02| < 1 - eps  (i.e., |cos(pitch)| != 0)
#     pitch_reg = tf.asin(tf.clip_by_value(r02, -one, one))
#     roll_reg = tf.math.atan2(-r12, r22)
#     yaw_reg = tf.math.atan2(-r01, r00)

#     # Gimbal lock: cos(pitch) ~ 0  -> pitch = ±pi/2
#     pitch_gl = (_tf_pi(dtype) / tf.cast(2.0, dtype)) * tf.sign(r02)
#     roll_gl = tf.zeros_like(pitch_gl)  # set roll = 0 by convention
#     # Both cases use same formula: yaw = atan2(r10, r11)
#     yaw_gl = tf.math.atan2(r10, r11)

#     # Blend by condition
#     cond = tf.less(tf.abs(r02), (one - eps_t))
#     roll = tf.where(cond, roll_reg, roll_gl)
#     pitch = tf.where(cond, pitch_reg, pitch_gl)
#     yaw = tf.where(cond, yaw_reg, yaw_gl)
#     return tf.stack([roll, pitch, yaw], axis=-1)


# def euler_diff(angles1, angles2, order="xyz", degrees=False):
#     """
#     Compute relative Euler angle difference: angles_rel such that
#         R(angles2) * R(angles_rel) = R(angles1)

#     Args:
#         angles1: (..., 3) tensor of Euler angles [roll, pitch, yaw] (extrinsic XYZ)
#         angles2: (..., 3) tensor of Euler angles [roll, pitch, yaw] (extrinsic XYZ)
#         order:   rotation order string (currently only "xyz" extrinsic is supported)
#         degrees: whether input/output are in degrees
#     Returns:
#         (..., 3) tensor of relative Euler angles [roll, pitch, yaw] (extrinsic XYZ)
#     """
#     if degrees:
#         angles1 = tf.math.multiply(angles1, _tf_pi(tf.float32) / 180.0)
#         angles2 = tf.math.multiply(angles2, _tf_pi(tf.float32) / 180.0)

#     # Build rotation matrices using extrinsic XYZ convention
#     R1 = _R_from_euler_xyz(angles1)
#     R2 = _R_from_euler_xyz(angles2)

#     # Compute relative rotation: Rrel = R2^T * R1
#     Rrel = tf.linalg.matmul(R2, R1, transpose_a=True)

#     # Extract Euler angles from Rrel using robust method with gimbal lock handling
#     out = _euler_xyz_from_R(Rrel)
#     if degrees:
#         out = tf.math.multiply(out, 180.0 / _tf_pi(tf.float32))
#     return out


def quat_to_euler(quat: np.ndarray) -> np.ndarray:
    return R.from_quat(quat).as_euler("xyz")  # extrinsic XYZ: roll, pitch, yaw


def _generate_examples(paths) -> Iterator[tuple[str, Any]]:
    """Yields episodes for list of trajectory folder paths."""

    def _parse_trajectory_file(traj_file_path):
        # Load HDF5 data
        with h5py.File(traj_file_path, "r") as f:
            # Load actions: joint_velocity (7) + gripper_position (1) = 8D
            cartesian_position = np.array(f["action/cartesian_position"])  # (T, 7)

            # cartesian_position = np.concatenate(
            #     [
            #         cartesian_position[:, :3],  # x, y, z
            #         quat_to_euler(cartesian_position[:, 3:7]),  # rx, ry, rz
            #     ],
            #     axis=-1,
            # )  # (T, 6)

            action_gripper = np.array(f["action/gripper_action"])  # (T,)
            images = np.array(f["observation/image/image"])  # (T, H, W, 3)
            wrist_images = np.array(f["observation/image/wrist_image"])  # (T, H, W, 3)

            obs_cartesian = np.array(f["observation/robot_state/cartesian_position"])  # (T, 6)
            skip_actions = np.array(f["observation/timestamp/skip_action"])  # (T,)

            num_timesteps = len(cartesian_position)

        # Extract video frames
        print(f"Extracting frames from {traj_file_path}...")
        # Build episode steps
        episode = []
        for i in range(num_timesteps):
            if skip_actions[i]:
                continue
            if i == 0:
                state = np.concatenate(
                    [
                        obs_cartesian[i],  # (6,)
                        [-1],  # gripper position
                    ]
                ).astype(np.float32)
            # Construct state: cartesian_position (3) + euler_angle (3) + gripper (1) = 7D
            else:
                state = np.concatenate(
                    [
                        obs_cartesian[i],  # (6,)
                        action_gripper[i - 1 : i],  # gripper position
                    ]
                ).astype(np.float32)

            action = np.concatenate(
                [
                    cartesian_position[i, :3],  # (7,)
                    quat_to_euler(cartesian_position[i, 3:7]),  # (6,)
                    action_gripper[i : i + 1],  # (1,)
                ]
            ).astype(np.float32)

            # Add step to episode
            episode.append(
                {
                    "observation": {
                        "image": images[i],
                        "wrist_image": wrist_images[i],
                        "state": state,
                    },
                    "action": action,
                    "discount": 1.0,
                    "reward": float(i == (num_timesteps - 1)),
                    "is_first": i == 0,
                    "is_last": i == (num_timesteps - 1),
                    "is_terminal": i == (num_timesteps - 1),
                    "language_instruction": "put marker in the bowl",
                }
            )

        # Create output data sample
        sample = {
            "steps": episode,
            "episode_metadata": {
                "file_path": str(traj_file_path),
            },
        }

        # Create unique key from folder name
        unique_key = str(traj_file_path)
        yield unique_key, sample

    # Parse examples from trajectory folder paths
    for traj_file_path in paths:
        try:
            yield from _parse_trajectory_file(traj_file_path)
        except Exception as e:
            # Log the error
            print(f"WARNING: Skipping trajectory file {traj_file_path}: {type(e).__name__}: {e!s}")
            continue


class FrankaDataset(MultiThreadedDatasetBuilder):
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
                                    "image": tfds.features.Image(
                                        shape=(224, 224, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Exterior camera 1 RGB observation (camera 38872458).",
                                    ),
                                    "wrist_image": tfds.features.Image(
                                        shape=(224, 224, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Wrist camera RGB observation (camera 10501775).",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc="Robot state: [cartesian_pos (3), euler_angle (3), gripper (1)].",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(7,),
                                dtype=np.float32,
                                doc="Robot action: [cartesian position (6), gripper (1)].",
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
                        }
                    ),
                }
            )
        )

    def _split_paths(self):
        """Define filepaths for data splits."""
        # Get trajectory root directory from env variable, default to 2025-11-18/
        traj_root = Path(os.getenv("TRAJECTORY_ROOT_DIR", "/home/irom-lab/new_camera_3"))

        if not traj_root.exists():
            raise FileNotFoundError(
                f"Trajectory root directory not found: {traj_root}\n"
                f"Please set TRAJECTORY_ROOT_DIR environment variable or update the default path."
            )

        # Get all .h5 files under trajectory root
        traj_files = list(traj_root.rglob("*.h5"))

        return {
            "train": traj_files,
        }
