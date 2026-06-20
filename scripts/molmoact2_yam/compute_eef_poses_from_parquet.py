#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
import json
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

import numpy as np
import pyarrow.parquet as pq


_TRUNC_RE = re.compile(
    r"Truncating episode\s+(?P<episode>\d+)\s+in\s+.*?/data/chunk-(?P<chunk>\d+)/file-(?P<file>\d+)\.parquet;"
    r"\s+dropping\s+(?P<drop>\d+)/"
)


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray


def parse_truncations(log_path: Path | None) -> dict[tuple[int, int, int], int]:
    truncations: dict[tuple[int, int, int], int] = {}
    if log_path is None:
        return truncations
    for line in log_path.read_text(errors="replace").splitlines():
        match = _TRUNC_RE.search(line)
        if not match:
            continue
        key = (int(match["chunk"]), int(match["file"]), int(match["episode"]))
        truncations[key] = int(match["drop"])
    return truncations


def parse_chunk_file(path: Path) -> tuple[int, int]:
    match = re.search(r"chunk-(\d+)/file-(\d+)\.parquet$", path.as_posix())
    if not match:
        raise ValueError(f"Could not parse chunk/file from {path}")
    return int(match.group(1)), int(match.group(2))


def _parse_floats(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(x) for x in text.split()], dtype=np.float64)


def parse_urdf(urdf_path: Path) -> dict[str, UrdfJoint]:
    root = ET.fromstring(urdf_path.read_text())
    joints: dict[str, UrdfJoint] = {}
    for joint_el in root.findall("joint"):
        name = joint_el.get("name")
        if not name:
            continue
        parent_el = joint_el.find("parent")
        child_el = joint_el.find("child")
        if parent_el is None or child_el is None:
            continue
        origin_el = joint_el.find("origin")
        if origin_el is not None:
            origin_xyz = _parse_floats(origin_el.get("xyz"), (0.0, 0.0, 0.0))
            origin_rpy = _parse_floats(origin_el.get("rpy"), (0.0, 0.0, 0.0))
        else:
            origin_xyz = np.zeros(3, dtype=np.float64)
            origin_rpy = np.zeros(3, dtype=np.float64)
        axis_el = joint_el.find("axis")
        axis = _parse_floats(axis_el.get("xyz") if axis_el is not None else None, (1.0, 0.0, 0.0))
        joints[name] = UrdfJoint(
            name=name,
            type=joint_el.get("type") or "fixed",
            parent=parent_el.get("link") or "",
            child=child_el.get("link") or "",
            origin_xyz=origin_xyz,
            origin_rpy=origin_rpy,
            axis=axis,
        )
    return joints


def build_chain(joints: dict[str, UrdfJoint], base_link: str, target_link: str) -> list[UrdfJoint]:
    child_to_joint = {joint.child: joint for joint in joints.values()}
    chain: list[UrdfJoint] = []
    current = target_link
    while current != base_link:
        if current not in child_to_joint:
            raise ValueError(f"Cannot reach {base_link!r} from {target_link!r}; stuck at {current!r}")
        joint = child_to_joint[current]
        chain.append(joint)
        current = joint.parent
    chain.reverse()
    return chain


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _axis_rotation_batch(axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    k = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    k2 = k @ k
    sin = np.sin(angles)[:, None, None]
    cos = np.cos(angles)[:, None, None]
    return np.eye(3) + sin * k + (1.0 - cos) * k2


def _matrix_to_rpy_batch(rot: np.ndarray) -> np.ndarray:
    sy = -rot[..., 2, 0]
    cy = np.sqrt(rot[..., 0, 0] ** 2 + rot[..., 1, 0] ** 2)
    near_singular = cy < 1e-6
    roll = np.where(
        near_singular,
        np.arctan2(-rot[..., 1, 2], rot[..., 1, 1]),
        np.arctan2(rot[..., 2, 1], rot[..., 2, 2]),
    )
    pitch = np.arctan2(sy, cy)
    yaw = np.where(
        near_singular,
        np.zeros_like(sy),
        np.arctan2(rot[..., 1, 0], rot[..., 0, 0]),
    )
    return np.stack([roll, pitch, yaw], axis=-1)


def _origin_transform(joint: UrdfJoint) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rpy_to_matrix(joint.origin_rpy)
    transform[:3, 3] = joint.origin_xyz
    return transform


def make_fk(urdf_path: Path, *, base_link: str, target_link: str):
    joints = parse_urdf(urdf_path)
    chain = build_chain(joints, base_link, target_link)
    print(f"FK chain {base_link} -> {target_link}: {[joint.name for joint in chain]}", flush=True)
    origin_transforms = [_origin_transform(joint) for joint in chain]

    def fk(joint_positions: np.ndarray) -> np.ndarray:
        joint_positions = np.asarray(joint_positions, dtype=np.float64)
        if joint_positions.ndim != 2 or joint_positions.shape[1] < 6:
            raise ValueError(f"Expected (T, >=6) joint positions, got {joint_positions.shape}")
        num_steps = joint_positions.shape[0]
        out = np.tile(np.eye(4, dtype=np.float64), (num_steps, 1, 1))
        joint_values = {f"joint{i + 1}": joint_positions[:, i] for i in range(6)}
        for joint, origin_transform in zip(chain, origin_transforms):
            out = out @ origin_transform
            if joint.type in {"revolute", "continuous"}:
                angles = joint_values.get(joint.name)
                if angles is not None:
                    joint_transform = np.tile(np.eye(4, dtype=np.float64), (num_steps, 1, 1))
                    joint_transform[:, :3, :3] = _axis_rotation_batch(joint.axis, angles)
                    out = out @ joint_transform
            elif joint.type == "prismatic":
                values = joint_values.get(joint.name)
                if values is not None:
                    joint_transform = np.tile(np.eye(4, dtype=np.float64), (num_steps, 1, 1))
                    joint_transform[:, :3, 3] = joint.axis[None, :] * values[:, None]
                    out = out @ joint_transform
        xyz = out[:, :3, 3]
        rpy = _matrix_to_rpy_batch(out[:, :3, :3])
        return np.concatenate([xyz, rpy], axis=-1).astype(np.float32)

    return fk


def load_columns(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["action", "observation.state", "episode_index", "frame_index"])
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
    episodes = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
    frame_indices = np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64)
    return actions, states, episodes, frame_indices


def tfds_file_path_for(raw_dir_for_keys: Path, chunk_idx: int, file_idx: int) -> str:
    return str(raw_dir_for_keys / "data" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet")


def process_dataset(
    *,
    raw_dir: Path,
    raw_dir_for_keys: Path,
    urdf_path: Path,
    base_link: str,
    target_link: str,
    truncations: dict[tuple[int, int, int], int],
    max_episodes: int | None,
    log_every: int,
) -> dict[str, np.ndarray]:
    parquet_files = sorted((raw_dir / "data").glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {raw_dir / 'data'}")

    fk = make_fk(urdf_path, base_link=base_link, target_link=target_link)
    episode_ids: list[str] = []
    episode_file_paths: list[str] = []
    episode_indices: list[int] = []
    chunk_indices: list[int] = []
    file_indices: list[int] = []
    first_frame_indices: list[int] = []
    episode_lengths: list[int] = []
    left_state_chunks: list[np.ndarray] = []
    right_state_chunks: list[np.ndarray] = []
    left_action_chunks: list[np.ndarray] = []
    right_action_chunks: list[np.ndarray] = []

    total_steps = 0
    started = time.time()
    for path_idx, path in enumerate(parquet_files, start=1):
        chunk_idx, file_idx = parse_chunk_file(path)
        actions, states, episodes, frame_indices = load_columns(path)
        for episode in np.unique(episodes):
            rows = np.flatnonzero(episodes == episode)
            drop = truncations.get((chunk_idx, file_idx, int(episode)), 0)
            if drop:
                rows = rows[:-drop]
            if rows.size == 0:
                continue

            ep_states = states[rows]
            ep_actions = actions[rows]
            left_state = fk(ep_states[:, :6])
            right_state = fk(ep_states[:, 7:13])
            left_action = fk(ep_actions[:, :6])
            right_action = fk(ep_actions[:, 7:13])

            tfds_file_path = tfds_file_path_for(raw_dir_for_keys, chunk_idx, file_idx)
            episode_file_paths.append(tfds_file_path)
            episode_ids.append(tfds_file_path)
            episode_indices.append(int(episode))
            chunk_indices.append(chunk_idx)
            file_indices.append(file_idx)
            first_frame_indices.append(int(frame_indices[rows[0]]))
            episode_lengths.append(int(rows.size))
            left_state_chunks.append(left_state)
            right_state_chunks.append(right_state)
            left_action_chunks.append(left_action)
            right_action_chunks.append(right_action)
            total_steps += int(rows.size)

            if max_episodes is not None and len(episode_ids) >= max_episodes:
                break
        if path_idx % log_every == 0 or (max_episodes is not None and len(episode_ids) >= max_episodes):
            elapsed = time.time() - started
            print(
                f"loaded {path_idx}/{len(parquet_files)} parquet files; "
                f"episodes={len(episode_ids)} steps={total_steps} elapsed={elapsed:.1f}s "
                f"rate={total_steps / max(elapsed, 1e-6):.0f} steps/s",
                flush=True,
            )
        if max_episodes is not None and len(episode_ids) >= max_episodes:
            break

    if not episode_ids:
        raise RuntimeError("No episodes produced")
    episode_lengths_arr = np.asarray(episode_lengths, dtype=np.int32)
    episode_starts = np.concatenate([[0], np.cumsum(episode_lengths_arr[:-1])]).astype(np.int64)
    return {
        "episode_ids": np.asarray(episode_ids),
        "episode_file_paths": np.asarray(episode_file_paths),
        "episode_indices": np.asarray(episode_indices, dtype=np.int64),
        "chunk_indices": np.asarray(chunk_indices, dtype=np.int32),
        "file_indices": np.asarray(file_indices, dtype=np.int32),
        "first_frame_indices": np.asarray(first_frame_indices, dtype=np.int64),
        "episode_starts": episode_starts,
        "episode_lengths": episode_lengths_arr,
        "left_eef_pose": np.concatenate(left_state_chunks, axis=0),
        "right_eef_pose": np.concatenate(right_state_chunks, axis=0),
        "left_action_eef_pose": np.concatenate(left_action_chunks, axis=0),
        "right_action_eef_pose": np.concatenate(right_action_chunks, axis=0),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "schema_version": 1,
                    "dataset_name": "molmoact2_yam_dataset",
                    "pose_layout": "[x, y, z, roll, pitch, yaw]",
                    "pose_frame": f"{base_link} frame for each individual YAM arm",
                    "urdf_target_link": target_link,
                    "joint_order": [
                        "left_joint1",
                        "left_joint2",
                        "left_joint3",
                        "left_joint4",
                        "left_joint5",
                        "left_joint6",
                        "left_gripper",
                        "right_joint1",
                        "right_joint2",
                        "right_joint3",
                        "right_joint4",
                        "right_joint5",
                        "right_joint6",
                        "right_gripper",
                    ],
                    "join_keys": [
                        "episode_metadata.file_path",
                        "episode_metadata.episode_index",
                        "episode_metadata.chunk_index",
                        "episode_metadata.file_index",
                    ],
                    "source": "YAM URDF from allenai/molmoact2 submodule Everloom-129/YAM at 9f06bba2a36dd84fb36d0c31337c85d4bf1cea22",
                },
                sort_keys=True,
            )
        ),
    }


def save_npz(arrays: dict[str, np.ndarray], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with io.BytesIO() as buf:
        np.savez_compressed(buf, **arrays)
        output_path.write_bytes(buf.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--raw-dir-for-keys",
        type=Path,
        default=None,
        help="Raw dir prefix used when forming episode_metadata.file_path keys. Defaults to --raw-dir.",
    )
    parser.add_argument(
        "--urdf-path",
        type=Path,
        default=Path(__file__).with_name("yam.urdf"),
    )
    parser.add_argument("--base-link", default="base_link")
    parser.add_argument("--target-link", default="link_6")
    parser.add_argument("--truncation-log", type=Path)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    truncations = parse_truncations(args.truncation_log)
    raw_dir_for_keys = args.raw_dir_for_keys or args.raw_dir
    print(f"raw_dir={args.raw_dir}")
    print(f"raw_dir_for_keys={raw_dir_for_keys}")
    print(f"urdf_path={args.urdf_path}")
    print(f"truncations={len(truncations)}")
    arrays = process_dataset(
        raw_dir=args.raw_dir,
        raw_dir_for_keys=raw_dir_for_keys,
        urdf_path=args.urdf_path,
        base_link=args.base_link,
        target_link=args.target_link,
        truncations=truncations,
        max_episodes=args.max_episodes,
        log_every=args.log_every,
    )
    save_npz(arrays, args.output)
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")
    print(
        f"episodes={len(arrays['episode_ids'])} steps={len(arrays['left_eef_pose'])} "
        f"left_shape={arrays['left_eef_pose'].shape} right_shape={arrays['right_eef_pose'].shape}"
    )


if __name__ == "__main__":
    main()
