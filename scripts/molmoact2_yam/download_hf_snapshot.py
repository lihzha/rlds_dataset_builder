#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi
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


def is_retryable_error(exc: BaseException) -> bool:
    retryable_names = {
        "ChunkedEncodingError",
        "ConnectionError",
        "ConnectionResetError",
        "ConnectTimeout",
        "ProtocolError",
        "ReadTimeout",
        "SSLError",
        "Timeout",
    }
    transient_fragments = (
        "429",
        "500",
        "502",
        "503",
        "504",
        "Connection aborted",
        "Connection reset",
        "Downloaded size mismatch",
        "EOF occurred",
        "ProtocolError",
        "Read timed out",
        "Remote end closed connection",
        "Too Many Requests",
    )

    current: BaseException | None = exc
    while current is not None:
        response = getattr(current, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code in {429, 500, 502, 503, 504}:
            return True
        if current.__class__.__name__ in retryable_names:
            return True
        message = str(current)
        if "No space left on device" in message:
            return False
        if any(fragment in message for fragment in transient_fragments):
            return True
        current = current.__cause__ or current.__context__
    return False


def download_with_retry(
    *,
    repo_id: str,
    filename: str,
    raw_dir: Path,
    retries: int,
    retry_sleep: float,
    expected_size: int | None = None,
    force: bool = False,
) -> None:
    target = raw_dir / filename
    if target.exists() and not force:
        local_size = target.stat().st_size
        if expected_size is None and local_size > 0:
            print(f"skipping existing {filename}", flush=True)
            return
        if expected_size is not None and local_size == expected_size:
            print(f"skipping existing {filename}", flush=True)
            return
        print(
            f"redownloading {filename}; local size {local_size} != expected {expected_size}",
            flush=True,
        )

    for attempt in range(retries + 1):
        try:
            if target.exists():
                local_size = target.stat().st_size
                if force or expected_size is None or local_size != expected_size:
                    target.unlink()
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                local_dir=str(raw_dir),
                force_download=force,
            )
            if not target.exists() or target.stat().st_size == 0:
                raise RuntimeError(f"Downloaded file is missing or empty: {filename}")
            if expected_size is not None and target.stat().st_size != expected_size:
                raise RuntimeError(
                    f"Downloaded size mismatch for {filename}: "
                    f"got {target.stat().st_size}, expected {expected_size}"
                )
            return
        except Exception as exc:  # noqa: BLE001 - preserve HF exception context in logs.
            if target.exists() and expected_size is not None and target.stat().st_size != expected_size:
                target.unlink()
            if attempt >= retries or not is_retryable_error(exc):
                raise
            sleep_s = retry_sleep * (2 ** min(attempt, 4))
            print(
                f"retryable download error for {filename}: {exc}; "
                f"sleeping {sleep_s:.1f}s before retry {attempt + 1}/{retries}",
                flush=True,
            )
            time.sleep(sleep_s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--full", action="store_true", help="Download the full snapshot.")
    parser.add_argument("--max-files", type=int, default=1, help="Number of file triplets per chunk for smoke data.")
    parser.add_argument("--chunks", default="0", help="Comma-separated chunk indices for smoke data.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.environ.get("HF_SNAPSHOT_MAX_WORKERS", "1")),
        help="Use snapshot_download with this worker count for full downloads. A value of 1 uses resumable sequential downloads.",
    )
    parser.add_argument("--retries", type=int, default=int(os.environ.get("HF_DOWNLOAD_RETRIES", "8")))
    parser.add_argument("--retry-sleep", type=float, default=float(os.environ.get("HF_DOWNLOAD_RETRY_SLEEP", "30")))
    parser.add_argument("--force", action="store_true", help="Re-check files even when a local completed file exists.")
    args = parser.parse_args()

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    if args.full:
        if args.max_workers <= 1:
            repo_info = HfApi().repo_info(args.repo_id, repo_type="dataset", files_metadata=True)
            files = [sibling.rfilename for sibling in repo_info.siblings]
            expected_sizes = {sibling.rfilename: sibling.size for sibling in repo_info.siblings}
            for idx, filename in enumerate(files, start=1):
                print(f"[{idx}/{len(files)}] downloading {filename}", flush=True)
                download_with_retry(
                    repo_id=args.repo_id,
                    filename=filename,
                    raw_dir=args.raw_dir,
                    retries=args.retries,
                    retry_sleep=args.retry_sleep,
                    expected_size=expected_sizes.get(filename),
                    force=args.force,
                )
            return
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(args.raw_dir),
            max_workers=args.max_workers,
        )
        return

    chunks = [int(x) for x in args.chunks.split(",") if x.strip()]
    repo_info = HfApi().repo_info(args.repo_id, repo_type="dataset", files_metadata=True)
    expected_sizes = {sibling.rfilename: sibling.size for sibling in repo_info.siblings}
    for pattern in sample_patterns(args.max_files, chunks):
        print(f"downloading {pattern}")
        download_with_retry(
            repo_id=args.repo_id,
            filename=pattern,
            raw_dir=args.raw_dir,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
            expected_size=expected_sizes.get(pattern),
            force=args.force,
        )


if __name__ == "__main__":
    main()
