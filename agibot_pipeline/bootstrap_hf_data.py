"""One-time bootstrap: download shared assets to scratch and emit tasks.json.

Downloads (to a shared scratch directory):
  - all task_info/task_<id>.json files (217 files, ~315 MB)
  - all proprio_stats/*.tar files (7 files, ~265 GB) and extracts them

Emits <base>/tasks.json — the manifest for the array submit script.
Each entry: {"task_id": str, "obs_tars": [hf_path, ...], "size_bytes": int}.

Usage:
  python -m agibot_pipeline.bootstrap_hf_data \
      --base /path/to/shared_scratch/agibot \
      [--workers 8] [--skip-proprio] [--skip-task-info]
"""

import argparse
import json
import os
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm


HF_REPO = "agibot-world/AgiBotWorld-Beta"
HF_REPO_TYPE = "dataset"


def list_repo(api: HfApi):
    info = api.repo_info(HF_REPO, repo_type=HF_REPO_TYPE, files_metadata=True)
    return info.siblings


def download_file(rfilename: str, base: Path) -> Path:
    """Download one file from HF to <base>/<rfilename>. Idempotent."""
    target = base / rfilename
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(
        repo_id=HF_REPO,
        filename=rfilename,
        repo_type=HF_REPO_TYPE,
        local_dir=str(base),
    )
    return Path(cached)


def extract_proprio_tar(tar_path: Path, dest_root: Path) -> None:
    """Extract proprio_stats tar into dest_root.

    Tar contents are <task_id>/<episode_id>/proprio_stats.h5 (relative paths).
    We extract straight under dest_root/proprio_stats/.
    Idempotent: if every member already exists, skip.
    """
    with tarfile.open(tar_path, "r") as tf:
        members = tf.getmembers()
        out_root = dest_root / "proprio_stats"
        # Quick idempotency check
        if all((out_root / m.name).exists() for m in members if not m.isdir()):
            return
        out_root.mkdir(parents=True, exist_ok=True)
        tf.extractall(out_root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path,
                    help="Shared-scratch base dir for the dataset.")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel HF downloads.")
    ap.add_argument("--skip-task-info", action="store_true")
    ap.add_argument("--skip-proprio", action="store_true")
    ap.add_argument("--keep-tars", action="store_true",
                    help="Keep proprio_stats tar files after extraction.")
    args = ap.parse_args()

    base: Path = args.base
    base.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    print(f"Listing files in {HF_REPO}...")
    siblings = list_repo(api)
    print(f"  {len(siblings)} files total.")

    task_info_files = [s for s in siblings if s.rfilename.startswith("task_info/")]
    proprio_files = [s for s in siblings if s.rfilename.startswith("proprio_stats/")]
    obs_files = [s for s in siblings if s.rfilename.startswith("observations/")]

    # --- task_info/ ---
    if not args.skip_task_info:
        print(f"Downloading {len(task_info_files)} task_info files...")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(download_file, s.rfilename, base) for s in task_info_files]
            for _ in tqdm(as_completed(futs), total=len(futs), desc="task_info"):
                pass

    # --- proprio_stats/ ---
    if not args.skip_proprio:
        print(f"Downloading {len(proprio_files)} proprio_stats tars (~265 GB)...")
        downloaded_tars = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(download_file, s.rfilename, base): s.rfilename
                for s in proprio_files
            }
            for f in tqdm(as_completed(futs), total=len(futs), desc="proprio dl"):
                downloaded_tars.append(f.result())

        print("Extracting proprio_stats tars...")
        for tar_path in tqdm(downloaded_tars, desc="proprio extract"):
            extract_proprio_tar(tar_path, base)
            if not args.keep_tars:
                tar_path.unlink(missing_ok=True)

    # --- emit tasks.json ---
    print("Building tasks.json manifest...")
    by_task = {}
    for s in obs_files:
        # observations/<task_id>/<ep_start>-<ep_end>.tar
        parts = s.rfilename.split("/")
        if len(parts) != 3 or not parts[2].endswith(".tar"):
            continue
        tid = parts[1]
        by_task.setdefault(tid, []).append({"path": s.rfilename, "size": s.size or 0})

    manifest = []
    for tid in sorted(by_task.keys(), key=int):
        tars = sorted(by_task[tid], key=lambda x: x["path"])
        manifest.append(
            {
                "task_id": tid,
                "obs_tars": [t["path"] for t in tars],
                "size_bytes": sum(t["size"] for t in tars),
            }
        )

    out = base / "tasks.json"
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {out}: {len(manifest)} tasks, "
          f"{sum(t['size_bytes'] for t in manifest)/1e12:.2f} TB obs total.")


if __name__ == "__main__":
    main()
