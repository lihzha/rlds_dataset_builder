from __future__ import annotations

from collections import defaultdict
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Iterator

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import tensorflow_datasets as tfds
from tensorflow_datasets.core.utils import gcs_utils as tfds_gcs_utils

from molmoact2_yam_dataset.conversion_utils import MultiThreadedDatasetBuilder

tfds_gcs_utils._is_gcs_disabled = True  # Avoid unauthenticated TFDS GCS metadata probes on clusters.


HF_REPO_ID = "allenai/MolmoAct2-BimanualYAM-Dataset"
IMAGE_SIZE = (224, 224)
SOURCE_IMAGE_SIZE = (640, 360)
VIDEO_KEYS = {
    "top_image": "observation.images.top",
    "left_image": "observation.images.left",
    "right_image": "observation.images.right",
}
_CITATION = """
@misc{fang2026molmoact2actionreasoningmodels,
  title={MolmoAct2: Action Reasoning Models for Real-world Deployment},
  author={Fang, Haoquan and Duan, Jiafei and Clay, Donovan and Wang, Sam and Liu, Shuo and Huang, Weikai and Tsai, Wei-Chuan and Chen, Shirui and Ren, Zhongzheng and Farhadi, Ali and Fox, Dieter and Krishna, Ranjay},
  year={2026},
  eprint={2605.02881},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
"""


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _raw_root() -> Path:
    raw_dir = os.environ.get("MOLMOACT2_YAM_RAW_DIR")
    if not raw_dir:
        raise ValueError(
            "MOLMOACT2_YAM_RAW_DIR must point to a local Hugging Face snapshot "
            f"of {HF_REPO_ID}."
        )
    root = Path(raw_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"MOLMOACT2_YAM_RAW_DIR does not exist: {root}")
    return root


def _parse_chunk_file(path: Path) -> tuple[int, int]:
    match = re.search(r"chunk-(\d+)/file-(\d+)\.parquet$", path.as_posix())
    if not match:
        raise ValueError(f"Cannot parse chunk/file index from {path}")
    return int(match.group(1)), int(match.group(2))


def _matching_video_path(raw_root: Path, video_key: str, chunk_idx: int, file_idx: int) -> Path:
    return raw_root / "videos" / video_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"


def _resize_with_pad(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    image = image.convert("RGB")
    src_w, src_h = image.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR
    resized = image.resize((new_w, new_h), resample=resample)
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
    canvas.paste(resized, offset)
    return canvas


def _encode_jpeg(image: Image.Image) -> bytes:
    out = BytesIO()
    image.save(out, format="JPEG", quality=90, optimize=False)
    return out.getvalue()


def _ffmpeg_path() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("ffmpeg is required to decode MolmoAct2 AV1 videos") from exc


def _decode_video_jpegs(
    video_path: Path,
    frame_indices: list[int],
    *,
    source_size: tuple[int, int] = SOURCE_IMAGE_SIZE,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> dict[int, bytes]:
    """Decode selected frames from a LeRobot MP4 using ffmpeg rawvideo stdout.

    Some released MolmoAct2 files have parquet rows beyond the end of one
    camera video. Return the successfully decoded prefix and let the caller
    drop incomplete episodes instead of failing the full build.
    """
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not frame_indices:
        return {}

    needed = set(int(i) for i in frame_indices)
    max_needed = max(needed)
    width, height = source_size
    frame_bytes = width * height * 3
    cmd = [
        _ffmpeg_path(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "1",
        "-i",
        str(video_path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None

    decoded: dict[int, bytes] = {}
    frame_idx = 0
    try:
        while frame_idx <= max_needed:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) == 0:
                break
            if len(buf) != frame_bytes:
                print(
                    f"Short frame while decoding {video_path}: got {len(buf)} bytes, expected {frame_bytes}"
                )
                break
            if frame_idx in needed:
                image = Image.frombytes("RGB", (width, height), buf)
                decoded[frame_idx] = _encode_jpeg(_resize_with_pad(image, image_size))
                if len(decoded) == len(needed):
                    break
            frame_idx += 1
    finally:
        proc.stdout.close()

    proc.terminate()
    stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr is not None else ""
    rc = proc.wait(timeout=30)
    if rc not in (0, 255) and len(decoded) != len(needed):
        print(f"ffmpeg stopped early for {video_path} with code {rc}: {stderr}")
    missing = sorted(needed.difference(decoded))
    if missing:
        print(f"Missing {len(missing)} decoded frames from {video_path}; first missing={missing[:5]}")
    return decoded


def _load_task_maps(raw_root: Path) -> tuple[dict[int, str], dict[int, str]]:
    tasks_path = raw_root / "meta" / "tasks.parquet"
    annotated_path = raw_root / "meta" / "tasks_annotated.parquet"
    task_by_index: dict[int, str] = {}
    annotated_by_episode: dict[int, str] = {}

    if tasks_path.exists():
        table = pq.read_table(tasks_path)
        cols = table.to_pydict()
        tasks = cols.get("task") or cols.get("tasks") or []
        if "task_index" in cols:
            indices = cols["task_index"]
        elif "index" in cols:
            indices = cols["index"]
        else:
            indices = list(range(len(tasks)))
        task_by_index = {int(i): str(t) for i, t in zip(indices, tasks) if t is not None}

    if annotated_path.exists():
        table = pq.read_table(annotated_path)
        cols = table.to_pydict()
        tasks = cols.get("task") or cols.get("tasks") or []
        if "episode_index" in cols:
            episode_indices = cols["episode_index"]
        elif "index" in cols:
            episode_indices = cols["index"]
        else:
            episode_indices = list(range(len(tasks)))
        annotated_by_episode = {
            int(i): str(t).strip()
            for i, t in zip(episode_indices, tasks)
            if t is not None and str(t).strip()
        }

    return task_by_index, annotated_by_episode


def _language_for(
    episode_index: int,
    task_index: int,
    task_by_index: dict[int, str],
    annotated_by_episode: dict[int, str],
) -> str:
    annotated = annotated_by_episode.get(int(episode_index))
    if annotated:
        return annotated
    return task_by_index.get(int(task_index), "")


def _read_column(table: pq.ParquetFile | Any, name: str) -> list[Any]:
    return table.column(name).to_pylist()


def _generate_examples(paths: list[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    raw_root = _raw_root()
    task_by_index, annotated_by_episode = _load_task_maps(raw_root)
    max_episodes = _env_int("MOLMOACT2_YAM_MAX_EPISODES")
    yielded = 0

    for parquet_name in paths:
        parquet_path = Path(parquet_name)
        chunk_idx, file_idx = _parse_chunk_file(parquet_path)
        table = pq.read_table(parquet_path)
        action = _read_column(table, "action")
        state = _read_column(table, "observation.state")
        episode_index = _read_column(table, "episode_index")
        task_index = _read_column(table, "task_index")
        frame_index = _read_column(table, "frame_index")

        num_rows = len(action)
        frame_positions = list(range(num_rows))
        frame_sets = {
            obs_key: _decode_video_jpegs(
                _matching_video_path(raw_root, video_key, chunk_idx, file_idx),
                frame_positions,
            )
            for obs_key, video_key in VIDEO_KEYS.items()
        }
        complete_frame_positions = set(frame_positions)
        for decoded_frames in frame_sets.values():
            complete_frame_positions.intersection_update(decoded_frames.keys())
        if not complete_frame_positions:
            print(f"Skipping {parquet_path}; no complete camera frames decoded")
            continue

        rows_by_episode: dict[int, list[int]] = defaultdict(list)
        for row_idx, ep_idx in enumerate(episode_index):
            rows_by_episode[int(ep_idx)].append(row_idx)

        for ep_idx in sorted(rows_by_episode):
            row_indices = rows_by_episode[ep_idx]
            if not row_indices:
                continue
            if any(row_idx not in complete_frame_positions for row_idx in row_indices):
                missing_count = sum(row_idx not in complete_frame_positions for row_idx in row_indices)
                print(
                    f"Skipping episode {int(ep_idx)} in {parquet_path}; "
                    f"{missing_count}/{len(row_indices)} rows lack a complete camera triplet"
                )
                continue
            first_row = row_indices[0]
            language_instruction = _language_for(
                ep_idx,
                int(task_index[first_row]),
                task_by_index,
                annotated_by_episode,
            )
            steps = []
            for step_idx, row_idx in enumerate(row_indices):
                is_last = step_idx == len(row_indices) - 1
                steps.append(
                    {
                        "observation": {
                            "top_image": frame_sets["top_image"][row_idx],
                            "left_image": frame_sets["left_image"][row_idx],
                            "right_image": frame_sets["right_image"][row_idx],
                            "state": np.asarray(state[row_idx], dtype=np.float32),
                        },
                        "action": np.asarray(action[row_idx], dtype=np.float32),
                        "discount": np.float32(1.0),
                        "reward": np.float32(1.0 if is_last else 0.0),
                        "is_first": step_idx == 0,
                        "is_last": is_last,
                        "is_terminal": is_last,
                        "language_instruction": language_instruction,
                    }
                )

            key = f"chunk-{chunk_idx:03d}_file-{file_idx:03d}_episode-{ep_idx:06d}"
            yield key, {
                "steps": steps,
                "episode_metadata": {
                    "file_path": str(parquet_path),
                    "episode_index": np.int64(ep_idx),
                    "chunk_index": np.int32(chunk_idx),
                    "file_index": np.int32(file_idx),
                    "num_steps": np.int32(len(row_indices)),
                    "first_frame_index": np.int64(frame_index[first_row]),
                    "language_instruction": language_instruction,
                },
            }
            yielded += 1
            if max_episodes is not None and yielded >= max_episodes:
                return


class Molmoact2YamDataset(MultiThreadedDatasetBuilder):
    """RLDS builder for the MolmoAct2 Bimanual YAM LeRobot dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial MolmoAct2 Bimanual YAM RLDS conversion."}
    N_WORKERS = int(os.environ.get("MOLMOACT2_YAM_N_WORKERS", "4"))
    MAX_PATHS_IN_MEMORY = int(os.environ.get("MOLMOACT2_YAM_MAX_PATHS_IN_MEMORY", "8"))
    PARSE_FCN = _generate_examples

    def _info(self) -> tfds.core.DatasetInfo:
        image = lambda doc: tfds.features.Image(
            shape=(IMAGE_SIZE[1], IMAGE_SIZE[0], 3),
            dtype=np.uint8,
            encoding_format="jpeg",
            doc=doc,
        )
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "top_image": image("Top RGB camera observation."),
                                    "left_image": image("Left arm RGB camera observation."),
                                    "right_image": image("Right arm RGB camera observation."),
                                    "state": tfds.features.Tensor(
                                        shape=(14,),
                                        dtype=np.float32,
                                        doc="Bimanual YAM joint state: left six joints, left gripper, right six joints, right gripper.",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(14,),
                                dtype=np.float32,
                                doc="Bimanual YAM absolute joint-position action with the same layout as state.",
                            ),
                            "discount": tfds.features.Scalar(dtype=np.float32),
                            "reward": tfds.features.Scalar(dtype=np.float32),
                            "is_first": tfds.features.Scalar(dtype=np.bool_),
                            "is_last": tfds.features.Scalar(dtype=np.bool_),
                            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                            "language_instruction": tfds.features.Text(),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(),
                            "episode_index": tfds.features.Scalar(dtype=np.int64),
                            "chunk_index": tfds.features.Scalar(dtype=np.int32),
                            "file_index": tfds.features.Scalar(dtype=np.int32),
                            "num_steps": tfds.features.Scalar(dtype=np.int32),
                            "first_frame_index": tfds.features.Scalar(dtype=np.int64),
                            "language_instruction": tfds.features.Text(),
                        }
                    ),
                }
            ),
            supervised_keys=None,
            homepage=f"https://huggingface.co/datasets/{HF_REPO_ID}",
            citation=_CITATION,
        )

    def _split_paths(self) -> dict[str, list[str]]:
        raw_root = _raw_root()
        with open(raw_root / "meta" / "info.json", "r") as f:
            info = json.load(f)
        expected_video_keys = set(info["features"]).intersection(VIDEO_KEYS.values())
        missing_video_keys = set(VIDEO_KEYS.values()).difference(expected_video_keys)
        if missing_video_keys:
            raise ValueError(f"Missing expected video keys in meta/info.json: {sorted(missing_video_keys)}")

        parquet_files = sorted((raw_root / "data").glob("chunk-*/file-*.parquet"))
        chunk_filter = os.environ.get("MOLMOACT2_YAM_CHUNKS")
        if chunk_filter:
            allowed = {int(x) for x in chunk_filter.split(",") if x.strip()}
            parquet_files = [p for p in parquet_files if _parse_chunk_file(p)[0] in allowed]

        usable = []
        for path in parquet_files:
            chunk_idx, file_idx = _parse_chunk_file(path)
            videos = [_matching_video_path(raw_root, key, chunk_idx, file_idx) for key in VIDEO_KEYS.values()]
            if all(v.exists() for v in videos):
                usable.append(str(path))
            else:
                missing = [str(v) for v in videos if not v.exists()]
                print(f"Skipping {path}; missing videos: {missing}")

        max_files = _env_int("MOLMOACT2_YAM_MAX_FILES")
        if max_files is not None:
            usable = usable[:max_files]
        if not usable:
            raise FileNotFoundError(f"No usable MolmoAct2-YAM parquet/video triplets under {raw_root}")
        print(f"MolmoAct2-YAM builder found {len(usable)} source parquet files")
        return {"train": usable}
