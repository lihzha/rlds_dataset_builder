#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download


REPO_ID = "allenai/MolmoAct2-BimanualYAM-Dataset"
VIDEO_KEYS = [
    "observation.images.top",
    "observation.images.left",
    "observation.images.right",
]


def sample_patterns(max_files: int, chunks: list[int]) -> list[str]:
    patterns = [
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.parquet",
        "meta/tasks_annotated.parquet",
        "meta/episodes/chunk-000/file-000.parquet",
    ]
    for chunk_idx in chunks:
        for file_idx in range(max_files):
            patterns.append(f"data/chunk-{chunk_idx:03d}/file-{file_idx:03d}.parquet")
            for video_key in VIDEO_KEYS:
                patterns.append(f"videos/{video_key}/chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4")
    return patterns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--full", action="store_true", help="Download the full snapshot.")
    parser.add_argument("--max-files", type=int, default=1, help="Number of file triplets per chunk for smoke data.")
    parser.add_argument("--chunks", default="0", help="Comma-separated chunk indices for smoke data.")
    args = parser.parse_args()

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    if args.full:
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(args.raw_dir),
        )
        return

    chunks = [int(x) for x in args.chunks.split(",") if x.strip()]
    for pattern in sample_patterns(args.max_files, chunks):
        print(f"downloading {pattern}")
        hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            filename=pattern,
            local_dir=str(args.raw_dir),
        )


if __name__ == "__main__":
    main()
